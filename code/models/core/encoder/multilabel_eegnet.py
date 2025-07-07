import torch
import warnings
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn.functional import elu
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import (
                                            MultilabelAccuracy, 
                                            MultilabelRecall, 
                                            MultilabelPrecision, 
                                            MultilabelF1Score, 
                                            MultilabelConfusionMatrix
                                        )
                                        
warnings.filterwarnings("ignore")

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
        
class MultilabelEEGNet(pl.LightningModule):
    def __init__(self, num_channels, time_steps, n_classes, lr=1e-3, one_cycle_lr=True, weight_decay=0.0, epochs=100, final_conv_length='auto', F1=8, D=2, kernel_length=64, pool_mode="mean", drop_prob=0.5, momentum=0.01, multi_label_threshold=0.5):
        super().__init__()
        
        if final_conv_length == "auto":
            assert time_steps is not None

        self.num_channels = num_channels
        self.time_steps = time_steps
        self.n_classes = n_classes
        self.final_conv_length = final_conv_length
        self.pool_mode = pool_mode
        self.F1 = F1
        self.D = D
        self.F2 = D * F1
        self.kernel_length = kernel_length
        self.drop_prob = drop_prob
        self.momentum = momentum
        self.one_cycle_lr = one_cycle_lr
        self.lr = lr
        self.weight_decay = weight_decay
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        self.epochs = epochs
        self.multi_label_threshold = multi_label_threshold

        # Input Transformation
        self.ensuredims = Ensure4d()
        self.dimshuffle = Expression(_transpose_to_b_1_c_0)

        # Temporal convolution
        self.conv_temporal = nn.Conv2d(in_channels=1, out_channels=self.F1, kernel_size=(1, self.kernel_length), stride=1, bias=False, padding=(0, self.kernel_length // 2))
        self.bnorm_temporal = nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3)

        # Spatial convolution
        self.conv_spatial = Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.num_channels, 1), max_norm=1, stride=1, bias=False, groups=self.F1, padding=(0, 0))
        self.bnorm_1 = nn.BatchNorm2d(self.F1 * self.D, momentum=self.momentum, affine=True, eps=1e-3)
        self.elu_1 = Expression(elu)
        self.pool_1 = pool_class(kernel_size=(1,4), stride=(1,4))
        self.drop_1 = nn.Dropout(p=self.drop_prob)

        # Separable convolution
        self.conv_separable_dept = nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 16), stride=1, bias=False, groups=self.F1 * self.D, padding=(0, 16 // 2))
        self.conv_separable_point = nn.Conv2d(self.F1 * self.D, self.F2, (1, 1), stride=1, bias=False, padding=(0, 0))
        self.bnorm_2 = nn.BatchNorm2d(self.F2, momentum=self.momentum, affine=True, eps=1e-3)
        self.elu_2 = Expression(elu)
        self.pool_2 = pool_class(kernel_size=(1, 8), stride=(1,8))
        self.drop_2 = nn.Dropout(p=self.drop_prob)

        with torch.no_grad():
            dummy_input = torch.ones((1, self.num_channels, self.time_steps, 1),dtype=torch.float32)
            features = self._feature_forward(dummy_input)

        n_out_virtual_chans = features.cpu().data.numpy().shape[2]
        
        if self.final_conv_length == "auto":
            n_out_time = features.cpu().data.numpy().shape[3]
            self.final_conv_length = n_out_time

        self.conv_classifier = nn.Conv2d(self.F2, self.n_classes, (n_out_virtual_chans, self.final_conv_length), bias=True)
        self.sigmoid = nn.Sigmoid()
        self.permute_back = Expression(_transpose_1_0)
        self.squeeze = Expression(squeeze_final_output)

        # Initialize weights
        _glorot_weight_zero_bias(self)
        self.save_hyperparameters()

        # Loss and Metrics
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.mean_train_loss = MeanMetric()
        self.mean_val_loss = MeanMetric()
        self.mean_test_loss = MeanMetric()
        self.mean_train_acc = MultilabelAccuracy(num_labels=n_classes, threshold=multi_label_threshold)
        self.mean_val_acc = MultilabelAccuracy(num_labels=n_classes, threshold=multi_label_threshold)
        self.mean_test_acc = MultilabelAccuracy(num_labels=n_classes, threshold=multi_label_threshold)
        self.mean_train_f1 = MultilabelF1Score(num_labels=n_classes, threshold=multi_label_threshold)
        self.mean_val_f1 = MultilabelF1Score(num_labels=n_classes, threshold=multi_label_threshold)
        self.mean_test_f1 = MultilabelF1Score(num_labels=n_classes, threshold=multi_label_threshold)

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
        x = self.conv_classifier(x)
        x = self.permute_back(x)
        x = self.squeeze(x)
        return x

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probs = self.sigmoid(logits)
            probs = probs.detach().cpu().numpy()
            preds = (probs > self.multi_label_threshold).astype(int)
            return preds
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.mean_train_loss(loss, weight=x.shape[0])
        self.mean_train_acc(logits, y)
        self.mean_train_f1(logits, y)

        self.log("train_batch_loss", self.mean_train_loss, prog_bar=True)
        self.log("train_batch_acc", self.mean_train_acc, prog_bar=True)
        self.log("train_batch_f1", self.mean_train_f1, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_loss", self.mean_train_loss, prog_bar=True)
        self.log("train_acc", self.mean_train_acc, prog_bar=True)
        self.log("train_f1", self.mean_train_f1, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.mean_val_loss(loss, weight=x.shape[0])
        self.mean_val_acc(logits, y)
        self.mean_val_f1(logits, y)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_loss", self.mean_val_loss, prog_bar=True)
        self.log("val_acc", self.mean_val_acc, prog_bar=True)
        self.log("val_f1", self.mean_val_f1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.mean_test_loss(loss, weight=x.shape[0])
        self.mean_test_acc(logits, y)
        self.mean_test_f1(logits, y)
        return loss

    def on_test_epoch_end(self):
        self.log("test_loss", self.mean_test_loss)
        self.log("test_acc", self.mean_test_acc)
        self.log("test_f1", self.mean_test_f1)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params = self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        if self.one_cycle_lr:
            one_cycle = torch.optim.lr_scheduler.OneCycleLR(
                optimizer = optimizer,
                max_lr = self.lr,
                total_steps = len(self.trainer.datamodule.train_dataloader()) * self.epochs,
                cycle_momentum = True
            )
            lr_scheduler = {
                "scheduler": one_cycle,
                "interval": "step",
                "name": "Learning Rate Scheduling"
            }
            return [optimizer], [lr_scheduler]
        else:
            return [optimizer]