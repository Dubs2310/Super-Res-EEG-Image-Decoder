import sys
import os
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.nn.functional import elu

sys.path.append(os.path.join(os.path.dirname(__file__), "..", '..'))
from utils.hdf5_data_split_generator import HDF5DataSplitGenerator
from utils.coco_data_handler import COCODataHandler

def _transpose_to_b_1_c_0(x):
    """Transpose input from [batch, channels, time, 1] to [batch, 1, channels, time]."""
    return x.permute(0, 3, 1, 2)


def _transpose_1_0(x):
    """Transpose dimensions 1 and 0 in the data."""
    return x.permute(0, 1, 3, 2)


def _glorot_weight_zero_bias(model):
    """Initialize parameters of all modules with glorot uniform/xavier initialization.
    
    Sets weights using glorot/xavier uniform initialization.
    Sets batch norm weights to 1 and biases to 0.
    
    Parameters
    ----------
    model: torch.nn.Module
        The model to initialize
    """
    for module in model.modules():
        if hasattr(module, "weight") and not "NLLLoss" in module.__class__.__name__:
            if not ("BatchNorm" in module.__class__.__name__):
                nn.init.xavier_uniform_(module.weight, gain=1)
            else:
                nn.init.constant_(module.weight, 1)
        if hasattr(module, "bias"):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


def squeeze_final_output(x):
    """Remove empty dimensions from the final output.
    
    Removes empty dimension at end and potentially removes empty time dimension.
    Does not use squeeze() as we never want to remove first dimension.
    
    Parameters
    ----------
    x: torch.Tensor
        Input tensor with shape [batch, features, 1, 1]
        
    Returns
    -------
    torch.Tensor
        Squeezed tensor with shape [batch, features]
    """
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x

class Conv2dWithConstraint(nn.Conv2d):
    """2D convolution with weight norm constraint.
    
    Normalizes weights to have a maximum L2 norm along dimension 0.
    
    Parameters
    ----------
    max_norm : float
        Maximum norm for weights
    """
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        # Apply weight normalization before forward pass
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)

class Expression(nn.Module):
    """Module that applies an arbitrary function on the forward pass.
    
    Parameters
    ----------
    expression_fn : callable
        Function to apply during forward pass
    """
    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if hasattr(self.expression_fn, "func") and hasattr(
            self.expression_fn, "kwargs"
        ):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__, str(self.expression_fn.kwargs)
            )
        elif hasattr(self.expression_fn, "__name__"):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr(self.expression_fn)
        return (
            self.__class__.__name__ +
            "(expression=%s) " % expression_str
        )

class Ensure4d(nn.Module):
    """Ensure input is 4D by adding singleton dimensions as needed."""
    def forward(self, x):
        while len(x.shape) < 4:
            x = x.unsqueeze(-1)
        return x

class EEGNet(nn.Module):
    def __init__(self, in_chans, input_window_samples, n_classes, 
                 final_conv_length='auto', F1=8, D=2, F2=16, 
                 kernel_length=64, drop_prob=0.5,
                 pool_class=nn.AvgPool2d, momentum=0.1
                 ):
        super().__init__()

        self.in_chans = in_chans
        self.input_window_samples = input_window_samples
        self.n_classes = n_classes
        self.final_conv_length = final_conv_length
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.drop_prob = drop_prob
        self.momentum = momentum

        # Input Transformation
        self.ensuredims = Ensure4d()
        self.dimshuffle = Expression(_transpose_to_b_1_c_0)

        # Temporal convolution
        self.conv_temporal = nn.Conv2d(
            in_channels=1,
            out_channels=self.F1,
            kernel_size=(1, self.kernel_length),
            stride=1,
            bias=False,
            padding=(0, self.kernel_length // 2)
        )
        self.bnorm_temporal = nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3)

        # Spatial convolution
        self.conv_spatial = Conv2dWithConstraint(
            self.F1,
            self.F1 * self.D,
            (self.in_chans, 1),
            max_norm=1,
            stride=1,
            bias=False,
            groups=self.F1,
            padding=(0, 0)
        )
        self.bnorm_1 = nn.BatchNorm2d(self.F1 * self.D, momentum=self.momentum, affine=True, eps=1e-3)
        self.elu_1 = Expression(elu)
        self.pool_1 = pool_class(kernel_size=(1,4), stride=(1,4))
        self.drop_1 = nn.Dropout(p=self.drop_prob)

        # Separable convolution
        self.conv_separable_dept = nn.Conv2d(
            self.F1 * self.D,
            self.F1 * self.D,
            (1, 16),
            stride=1,
            bias=False,
            groups=self.F1 * self.D,
            padding=(0,8)
        )
        self.conv_separable_point = nn.Conv2d(
            self.F1 * self.D,
            self.F2,
            (1, 1),
            stride=1,
            bias=False,
            padding=(0, 0)
        )
        self.bnorm_2 = nn.BatchNorm2d(self.F2, momentum=self.momentum, affine=True, eps=1e-3)
        self.elu_2 = Expression(elu)
        self.pool_2 = pool_class(kernel_size=(1, 8), stride=(1,8))
        self.drop_2 = nn.Dropout(p=self.drop_prob)

        with torch.no_grad():
            dumy = torch.ones((1, self.in_chans, self.input_window_samples, 1), dtype=torch.float32)
            out = self._feature_forward(dumy)
        
        n_out_virtual_chans = out.shape[2]
        n_out_time = out.shape[3]
        self.final_conv_length = n_out_time if final_conv_length == 'auto' else final_conv_length

        self.flat_dim = self.F2 * n_out_time

        # Embedding Layer 
        self.flatten = nn.Flatten()
        self.eeg_to_clip = nn.Linear(self.flat_dim, 512)

        # Classifier
        self.classifier = nn.Linear(512, self.n_classes)
        
        _glorot_weight_zero_bias(self)

    def _feature_forward(self, x):
        x = self.ensuredims(x)
        x = self.dimshuffle(x)

        x = self.conv_temporal(x)
        x = self.bnorm_temporal(x)
        
        x = self.conv_spatial(x)
        x = self.bnorm_1(x)
        x = self.elu_1(x)
        x = self.pool_1(x)
        x = self.drop_1(x)

        x = self.conv_separable_dept(x)
        x = self.conv_separable_point(x)
        x = self.bnorm_2(x)
        x = self.elu_2(x)
        x = self.pool_2(x)
        x = self.drop_2(x)

        return x
    
    def forward(self, x):
        x = self._feature_forward(x)
        x = self.flatten(x)
        clip_embed = self.eeg_to_clip(x)
        logits = self.classifier(clip_embed)

        return logits
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = self._feature_forward(x)
            x = self.flatten(x)
            clip_embed = self.eeg_to_clip(x)
            logits = self.classifier(clip_embed)
            probs = torch.sigmoid(logits)
        return probs
    
    def fit(self, generator, epochs, device, lr=1e-4, res='lo_res'): 
        """
            X is zip(metadata, eeg)
        """
        self.to(device)
        self.train()

        # mse_loss_fn = nn.MSELoss()
        bce_loss_fn = nn.BCEWithLogitsLoss()

        # Parameters for encoder vs classifier
        # enc_params = [p for name, p in self.named_parameters() if 'classifier' not in name]
        classifier_params = [p for name, p in self.named_parameters() if 'classifier' in name]

        # optimizer_encoder = torch.optim.Adam(enc_params, lr=lr)
        optimizer_classifier = torch.optim.Adam(classifier_params, lr=lr)

        for epoch in range(1, epochs + 1):
            # total_mse = 0.0
            total_cls_loss = 0.0

            progress_bar = tqdm(generator, desc=f"Epoch {epoch}/{epochs}", leave=True)
            for batch in progress_bar:
                lo_res = batch[res]
                one_hot_encoding = batch['one_hot_encoding']
                
                for i in range(len(lo_res)):
                    X, y = lo_res[i], one_hot_encoding[i]
                    X = X.unsqueeze(0).to(device)
                    y = y.unsqueeze(0).to(device)
                    # image_embed = embeds.unsqueeze(0).to(device)

                    logits = self(X)
                    print(logits)
                    print(y)

                    # # Training Encoder
                    # for param in classifier_params:
                    #     param.requires_grad = False

                    # optimizer_encoder.zero_grad()
                    # embed_loss = mse_loss_fn(clip_embed, image_embed)
                    # embed_loss.backward()
                    # optimizer_encoder.step()

                    # Training Classifier
                    # for param in enc_params:
                        # param.requires_grad = True

                    # for param in classifier_params:
                    #     param.requires_grad = True

                    optimizer_classifier.zero_grad()
                    cls_loss = bce_loss_fn(logits, y)
                    cls_loss.backward()
                    optimizer_classifier.step()

                    # # Reset all Params
                    # for params in self.parameters():
                    #     params.requires_grad = True

                    total_cls_loss += cls_loss.item()
                    # total_mse += embed_loss.item()
            
            # avg_mse = total_mse / len(generator.batch_size)
            avg_cls_loss = total_cls_loss / len(generator.batch_size)

            print(f'Epoch {epoch}: Classification Loss: {avg_cls_loss:.4f}')

if __name__ == '__main__':
    coco_data = COCODataHandler.get_instance()

    from torch.utils.data import DataLoader
    
    train_dataset = HDF5DataSplitGenerator(
        dataset_type="train",
        dataset_split="70/25/5",
        eeg_epoch_mode="around_evoked_event",
        fixed_length_duration=3,
        duration_before_onset=0.05,
        duration_after_onset=0.6
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    n_classes = len(coco_data.category_index.keys())
    in_chns, input_window_samples = train_loader.dataset[0]['lo_res'].shape

    eegNet = EEGNet(in_chns, input_window_samples, n_classes)
    eegNet.fit(train_loader, 1, 'cuda')