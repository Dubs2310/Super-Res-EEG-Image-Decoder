import os
import yaml
import math
import torch
import wandb
import argparse
import datetime
import importlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Callable
import torch.nn as nn, optim
from functools import partial
from packaging import version
import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch.nn.functional as F
from torch.nn.functional import elu
from dc_ldm.ldm_for_eeg import eLDM
from einops import rearrange, repeat
from contextlib import contextmanager
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from torchmetrics.functional import accuracy
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import Accuracy
from pytorch_lightning.loggers import WandbLogger
from skimage.metrics import structural_similarity as ssim
from torchvision.models import ViT_H_14_Weights, vit_h_14
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


class base_model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.acc = Accuracy(task="multiclass", num_classes=20)
        self.config = None
        self.save_hyperparameters()
        self.one_cycle_lr = True
        self.predictions = []
        self.ground_truth = []

    def training_step(self, batch, batch_idx):
        x,y = batch
        logit = self.forward(x.float()) 
        train_loss = self.loss(logit, y)
        _, y_pred = torch.max(logit, dim = 1)
        self.log("train_loss", train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", self.acc(y_pred, y), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return train_loss
        
    def validation_step(self, batch, batch_idx):
        x,y = batch
        logit = self.forward(x.float()) 
        val_loss = self.loss(logit, y)
        _, y_pred = torch.max(logit, dim = 1)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", self.acc(y_pred, y), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x,y = batch
        print("TEST")
        logit = self.forward(x.float())
        test_loss = self.loss(logit, y)
        _, y_pred = torch.max(logit, dim = 1)
        self.predictions.append(y_pred)
        self.ground_truth.append(y)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc", self.acc(y_pred, y), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return test_loss


class EEGNet_Embedding(base_model):
    def __init__(self, lr = 1e-3, one_cycle_lr = True, weight_decay = 0.0, epochs = 100, in_chans = 8, n_classes = 20, final_conv_length="auto", input_window_samples=None, F1=8, D=2, pool_mode="mean", kernel_length=64, drop_prob=0.25, momentum = 0.01, **kwargs):
        super().__init__()
        if final_conv_length == "auto":
            assert input_window_samples is not None
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_window_samples = input_window_samples
        self.final_conv_length = final_conv_length
        self.pool_mode = pool_mode
        self.F1 = F1
        self.D = D
        self.F2 = D*F1
        self.kernel_length = kernel_length
        self.drop_prob = drop_prob
        self.momentum = momentum
        self.one_cycle_lr = one_cycle_lr
        self.lr = lr
        self.weight_decay = weight_decay
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        self.loss = nn.NLLLoss() #EEGNet uses a log softmax layer. Therefore, using a NLLLoss() equates to CrossEntropyLoss()
        self.epochs = epochs

        self.ensuredims = Ensure4d() # from [b c t] to [b c t 1]
        self.dimshuffle = Expression(_transpose_to_b_1_c_0) # from [b c t 1] to [b 1 c t] --> first conv over temporal dim with kernel_length
        self.conv_temporal = nn.Conv2d(in_channels = 1, out_channels = self.F1, kernel_size = (1, self.kernel_length), stride = 1, bias = False, padding = (0, self.kernel_length // 2))
        self.bnorm_temporal = nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3)
        self.conv_spatial = Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.in_chans, 1), max_norm=1, stride=1, bias=False, groups=self.F1, padding=(0, 0))
        self.bnorm_1 = nn.BatchNorm2d(self.F1 * self.D, momentum=self.momentum, affine=True, eps=1e-3)
        self.elu_1 = Expression(elu)
        self.pool_1 = pool_class(kernel_size=(1, 4), stride=(1, 4))
        self.drop_1 = nn.Dropout(p=self.drop_prob)
        self.conv_separable_depth = nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 16), stride=1, bias=False, groups=self.F1 * self.D, padding=(0, 16 // 2))
        self.conv_separable_point = nn.Conv2d(self.F1 * self.D, self.F2, (1, 1), stride=1, bias=False, padding=(0, 0), )
        self.bnorm_2 = nn.BatchNorm2d(self.F2, momentum=self.momentum, affine=True, eps=1e-3)
        self.elu_2 = Expression(elu)
        self.pool_2 = pool_class(kernel_size=(1, 8), stride=(1, 8))
        self.drop_2 = nn.Dropout(p=self.drop_prob)

        #The following tests the output dimensions to pass the right dimensions to the classifier head
        out = self.partial_forward(torch.ones((1, self.in_chans, self.input_window_samples, 1), dtype=torch.float32))
        n_out_virtual_chans = out.cpu().data.numpy().shape[2]
        if self.final_conv_length == "auto":
            n_out_time = out.cpu().data.numpy().shape[3]
            self.final_conv_length = n_out_time
        self.embedding = nn.Linear(self.F2*15, 512)
        self.classifier = nn.Linear(512, self.n_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        # Transpose time back in third dimension (axis=2)
        self.permute_back = Expression(_transpose_1_0)
        self.squeeze = Expression(squeeze_final_output)
        _glorot_weight_zero_bias(self) #Initialize weights

    def partial_forward(self, x): #for a sample of [1 8 500] and kernel_size=128
        #Used to initially determine the input dimensions to the classifier head
        x = self.ensuredims(x) # [1 8 500 1]
        x = self.dimshuffle(x) # [1 1 8 500]
        x = self.conv_temporal(x) # [1 8 8 501]
        x = self.bnorm_temporal(x) # [1 8 8 501]
        x = self.elu_1(x) # [1 8 8 501]
        x = self.conv_spatial(x) # [1 16 1 501]
        x = self.bnorm_1(x) # [1 16 1 501]
        x = self.elu_1(x) # [1 16 1 501]
        x = self.pool_1(x) # [1 16 1 125]
        x = self.drop_1(x) # [1 16 1 125]
        x = self.conv_separable_depth(x) # [1 16 1 126]
        x = self.conv_separable_point(x) # [1 16 1 126]
        x = self.bnorm_2(x) # [1 16 1 126]
        x = self.elu_2(x) # [1 16 1 126]
        x = self.pool_2(x) # [1 16 1 15]
        x = self.drop_2(x) # [1 16 1 15]
        return x

    def forward(self, x):
        x = self.partial_forward(x) # [1 16 1 15] #bs x F2 x 1 x 15 (128*15)
        x = x.flatten(start_dim=1) # bs x F2*15
        x = self.embedding(x) # bs x 1024
        x = self.classifier(x) # bs x 20
        x = self.softmax(x) # [1 20 1 1]
        return x
    
    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay = self.weight_decay)
    #     return optimizer
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params = self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        if self.one_cycle_lr:
            one_cycle = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer = optimizer,
                    max_lr = self.lr,
                    total_steps = len(self.trainer.datamodule.train_dataloader()) * self.epochs,
                    # epochs = self.epochs,
                    # steps_per_epoch = self.trainer.estimated_stepping_batches // self.epochs,
                    cycle_momentum = True
                    )
            
            lr_scheduler = {
                "scheduler": one_cycle, #lr
                "interval": "step",
                "name": "Learning Rate Scheduling"
            }
            return [optimizer], [lr_scheduler]
        else:
            return [optimizer]
      

#Helper functions and classes
def _transpose_to_b_1_c_0(x):
    return x.permute(0, 3, 1, 2)


def _transpose_1_0(x):
    return x.permute(0, 1, 3, 2)


def _glorot_weight_zero_bias(model):
    """Initalize parameters of all modules by initializing weights with
    glorot
     uniform/xavier initialization, and setting biases to zero. Weights from
     batch norm layers are set to 1.

    Parameters
    ----------
    model: Module
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
    """Removes empty dimension at end and potentially removes empty time
     dimension. It does  not just use squeeze as we never want to remove
     first dimension.

    Returns
    -------
    x: torch.Tensor
        squeezed tensor
    """
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x


class Conv2dWithConstraint(nn.Conv2d):
    """Conv2d with weight constraint."""
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)
    

class Expression(nn.Module):
    """Compute given expression on forward pass.

    Parameters
    ----------
    expression_fn : callable
        Should accept variable number of objects of type
        `torch.autograd.Variable` to compute its output.
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
    def forward(self, x):
        while len(x.shape) < 4:
            x = x.unsqueeze(-1)
        return x


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class LitEma(nn.Module):
    def __init__(self, model, decay=0.9999, use_num_upates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        self.m_name2s_name = {}
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        self.register_buffer('num_updates', torch.tensor(0,dtype=torch.int) if use_num_upates else torch.tensor(-1,dtype=torch.int))

        for name, p in model.named_parameters():
            if p.requires_grad:
                #remove as '.'-character is not allowed in buffers
                s_name = name.replace('.','')
                self.m_name2s_name.update({name:s_name})
                self.register_buffer(s_name,p.clone().detach().data)

        self.collected_params = []

    def forward(self,model):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay,(1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
                else:
                    assert not key in self.m_name2s_name

    def copy_to(self, model):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                assert not key in self.m_name2s_name

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

# pytorch_diffusion + derived encoder decoder

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        h_ = x #(3,512,64,64) 3 = bs
        h_ = self.norm(h_)
        q = self.q(h_) #(3,512,64,64)
        k = self.k(h_) #(3,512,64,64)
        v = self.v(h_) #(3,512,64,64)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1).contiguous()   # b,hw,c (3,4096,512)
        k = k.reshape(b,c,h*w) # b,c,hw (3,512, 4096)
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j] #batch matrix-matrix product (3,4096,4096)
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1).contiguous()   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w) #(3,512,64,64)

        h_ = self.proj_out(h_)#(3,512,64,64)

        return x+h_


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)

#This is the LDM Encoder (not ours)
class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks, attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels, resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla", **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, 2*z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


#Used to turn de denoised latents into output images (from bs,3,64,64 to bs,3,256,256)
class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks, attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels, resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False, attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h



class PLMSSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda") and torch.cuda.is_available():
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        if ddim_eta != 0:
            raise ValueError('ddim_eta must be 0 for PLMS')
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps, num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(), ddim_timesteps=self.ddim_timesteps, eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt((1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self, S, batch_size, shape, conditioning=None, callback=None, normals_sequence=None, img_callback=None, quantize_x0=False, eta=0., mask=None, x0=None, temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None, verbose=True, x_T=None, log_every_t=100, unconditional_guidance_scale=1., unconditional_conditioning=None, **kwargs): # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for PLMS sampling is {size}')

        samples, intermediates = self.plms_sampling(conditioning, size, callback=callback, img_callback=img_callback, quantize_denoised=quantize_x0, mask=mask, x0=x0, ddim_use_original_steps=False, noise_dropout=noise_dropout, temperature=temperature, score_corrector=score_corrector, corrector_kwargs=corrector_kwargs, x_T=x_T, log_every_t=log_every_t, unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=unconditional_conditioning, **kwargs )
        return samples, intermediates

    @torch.no_grad()
    def plms_sampling(self, cond, shape, x_T=None, ddim_use_original_steps=False, callback=None, timesteps=None, quantize_denoised=False, mask=None, x0=None, img_callback=None, log_every_t=100, temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None, unconditional_guidance_scale=1., unconditional_conditioning=None, generator=None):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device, generator=generator)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = list(reversed(range(0,timesteps))) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running PLMS Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='PLMS Sampler', total=total_steps)
        old_eps = []

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_plms(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps, quantize_denoised=quantize_denoised, temperature=temperature, noise_dropout=noise_dropout, score_corrector=score_corrector, corrector_kwargs=corrector_kwargs, unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=unconditional_conditioning, old_eps=old_eps, t_next=ts_next)
            img, pred_x0, e_t = outs
            old_eps.append(e_t)
            if len(old_eps) >= 4:
                old_eps.pop(0)
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_plms(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False, temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None, unconditional_guidance_scale=1., unconditional_conditioning=None, old_eps=None, t_next=None):
        b, *_, device = *x.shape, x.device

        def get_model_output(x, t):
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(x, t, c)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            if score_corrector is not None:
                assert self.model.parameterization == "eps"
                e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

            return e_t

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

        def get_x_prev_and_pred_x0(e_t, index):
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

        e_t = get_model_output(x, t)
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            e_t_next = get_model_output(x_prev, t_next)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        return x_prev, pred_x0, e_t


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")

    reload = False
    module, cls = config["target"].rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)(**config.get("params", dict()))


class eLDM:

    def __init__(self,  EEGNet_config_path, EEGNet_ckpt_path, device=torch.device('cpu'), pretrain_root='../pretrains/ldm/label2img', logger=None,  ddim_steps=250,  global_pool=True,  use_time_cond=True ):
        self.ckp_path = os.path.join(pretrain_root, 'model.ckpt') #path to pretrained LDM
        self.config_path = os.path.join(pretrain_root, 'config.yaml') #path to pretrained LDM config
        config = OmegaConf.load(self.config_path)
        config.model.params.unet_config.params.use_time_cond = use_time_cond
        config.model.params.unet_config.params.global_pool = global_pool

        self.cond_dim = config.model.params.unet_config.params.context_dim #512

        model = instantiate_from_config(config.model)
        pl_sd = torch.load(self.ckp_path, map_location="cpu")['state_dict']
       
        m, u = model.load_state_dict(pl_sd, strict=False)
        model.cond_stage_trainable = True
        #3. Exchange cond stage model with our encoder
        model.cond_stage_model = cond_stage_model(EEGNet_config_path, EEGNet_ckpt_path) #Use EEGNet+ as condition stage model

        model.ddim_steps = ddim_steps
        model.re_init_ema()
        if logger is not None:
            logger.watch(model, log="all", log_graph=False)

        model.p_channels = config.model.params.channels
        model.p_image_size = config.model.params.image_size
        model.ch_mult = config.model.params.first_stage_config.params.ddconfig.ch_mult

        self.device = device    
        self.model = model
        self.ldm_config = config
        self.pretrain_root = pretrain_root

    def finetune(self, trainers, dataset, test_dataset, bs1, lr1, output_path, config=None):
        config.trainer = None
        config.logger = None
        self.model.main_config = config
        self.model.output_path = output_path
        # self.model.train_dataset = dataset
        self.model.run_full_validation_threshold = 0.15
        # stage one: train the cond encoder with the pretrained one
      
        # stage one: only optimize conditional encoders
        print('\n##### Stage One: only optimize conditional encoders #####')
        dataloader = DataLoader(dataset, batch_size=bs1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        self.model.unfreeze_whole_model()
        self.model.freeze_first_stage()

        self.model.learning_rate = lr1
        self.model.train_cond_stage_only = True
        self.model.eval_avg = config.eval_avg
        trainers.fit(self.model, dataloader, val_dataloaders=test_loader)

        self.model.unfreeze_whole_model()
        torch.save({ 'model_state_dict': self.model.state_dict(), 'config': config, 'state': torch.random.get_rng_state() }, os.path.join(output_path, 'checkpoint.pth'))
        

    @torch.no_grad()
    def generate(self, eeg_embedding, num_samples, ddim_steps, HW=None, limit=None, state=None):
        all_samples = []
        if HW is None:
            shape = (self.ldm_config.model.params.channels, self.ldm_config.model.params.image_size, self.ldm_config.model.params.image_size)
        else:
            num_resolutions = len(self.ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult)
            shape = (self.ldm_config.model.params.channels, HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        model = self.model.to(self.device)
        sampler = PLMSSampler(model)
        # sampler = DDIMSampler(model)
        if state is not None:
            torch.cuda.set_rng_state(state)
            
        with model.ema_scope():
            model.eval()
            for count, item in enumerate(eeg_embedding):
                if limit is not None:
                    if count >= limit:
                        break
                latent = item['eeg']
                gt_image = rearrange(item['image'], 'h w c -> 1 c h w') # h w c
                print(f"rendering {num_samples} examples in {ddim_steps} steps.")
                
                c = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                samples_ddim, _ = sampler.sample(S=ddim_steps, conditioning=c, batch_size=num_samples, shape=shape, verbose=False)
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                gt_image = torch.clamp((gt_image+1.0)/2.0, min=0.0, max=1.0)
                
                all_samples.append(torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0)) # put groundtruth at first
                
        
        # display as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples+1)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        model = model.to('cpu')
        
        return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8)


#Modifications
def read_model_config(config_path: str):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        config = config["parameters"]["model"]["parameters"]
        model_config = {key: config[key]["value"] for key in config}
    return model_config

#Update forward to discard classifier head
def updated_forward(self, x):
    x = self.partial_forward(x) # [1 16 1 15] #bs x F2 x 1 x 15 (128*15)
    x = x.flatten(start_dim=1) # bs x F2*15
    x = self.embedding(x) # bs x 512
    return x

class cond_stage_model(nn.Module):
    def __init__(self, EEGNet_config_path, EEGNet_ckpt_path):
        super().__init__()
        # prepare pretrained EEGNet
        model_config = read_model_config(config_path = EEGNet_config_path) #"C:/Users/s_gue/Desktop/master_project/sven-thesis/pytorch/EEG_encoder_setup/P001_model_config.yaml")
        model = EEGNet_Embedding(**model_config) #This is the EEGNet model mapping from EEG input to embedding
        checkpoint = torch.load(EEGNet_ckpt_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.forward = updated_forward.__get__(model)
        self.eegnet = model

        #Explanation: The output of the EEGNet is batch_size x 1 x 512. We need to map to the expected context size of the LDM
        #Therefore we use a 1x1 convolution that maps from depth 1 to 77
        #Note: As the EEGNet outputs a 512 dimensional vector, we avoid an extra linear mapping from 1024 to 512 (which is 0,5 million params!)
        self.channel_mapper = nn.Sequential(nn.Conv1d(1, 77, 1, bias=True))

    def forward(self, x):
        latent_crossattn = self.eegnet(x)
        return self.channel_mapper(latent_crossattn.unsqueeze(1))


class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer in taming, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random", sane_index_shape=False, legacy=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

class VQModel(pl.LightningModule):
    def __init__(self, ddconfig, lossconfig, n_embed, embed_dim, ckpt_path=None, ignore_keys=[], image_key="image", colorize_nlabels=None, monitor=None, batch_resize_range=None, scheduler_config=None, lr_g_factor=1.0, remap=None, sane_index_shape=False, use_ema=False): # tell vector quantizer to return indices as bhw 
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_,_,ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        if self.batch_resize_range is not None:
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            if self.global_step <= 4:
                # do the first few batches with max size to avoid later oom
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size+16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode="bicubic")
            x = x.detach()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step, last_layer=self.get_last_layer(), split="train", predicted_indices=ind)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=False, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step, last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=False, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step, last_layer=self.get_last_layer(), split="val"+suffix, predicted_indices=ind )

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step, last_layer=self.get_last_layer(), split="val"+suffix, predicted_indices=ind )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+ list(self.decoder.parameters())+ list(self.quantize.parameters())+ list(self.quant_conv.parameters())+ list(self.post_quant_conv.parameters()), lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                { 'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule), 'interval': 'step', 'frequency': 1 },
                { 'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule), 'interval': 'step', 'frequency': 1 }
            ]
            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


#Encodes the Image to the Latent
class VQModelInterface(VQModel):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    
class AutoencoderKL(pl.LightningModule):
    def __init__(self, ddconfig, lossconfig, embed_dim, ckpt_path=None, ignore_keys=[], image_key="image", colorize_nlabels=None, monitor=None,):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.trainable = False
        
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step, last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=False, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step, last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=False, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step, last_layer=self.get_last_layer(), split="val")
        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step, last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+ list(self.decoder.parameters())+ list(self.quant_conv.parameters())+ list(self.post_quant_conv.parameters()), lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x


def larger_the_better(gt, comp):
    return gt > comp


def smaller_the_better(gt, comp):
    return gt < comp


def mse_metric(img1, img2):
    return (np.square(img1 - img2)).mean()


def pcc_metric(img1, img2):
    return np.corrcoef(img1.reshape(-1), img2.reshape(-1))[0, 1]


def ssim_metric(img1, img2):
    return ssim(img1, img2, data_range=255, channel_axis=-1)


class psm_wrapper:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(self.device)

    @torch.no_grad()
    def __call__(self, img1, img2):
        if img1.shape[-1] == 3:
            img1 = rearrange(img1, 'w h c -> c w h')
            img2 = rearrange(img2, 'w h c -> c w h')
        img1 = img1 / 127.5 - 1.0
        img2 = img2 / 127.5 - 1.0
        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)
        return self.lpips(torch.FloatTensor(img1).to(self.device), torch.FloatTensor(img2).to(self.device)).item()


class fid_wrapper:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fid = FrechetInceptionDistance(feature=64)

    @torch.no_grad()
    def __call__(self, pred_imgs, gt_imgs):
        self.fid.reset()
        self.fid.update(torch.tensor(rearrange(gt_imgs, 'n w h c -> n c w h')), real=True)
        self.fid.update(torch.tensor(rearrange(pred_imgs, 'n w h c -> n c w h')), real=False)
        return self.fid.compute().item() 


def pair_wise_score(pred_imgs, gt_imgs, metric, is_sucess):
    # pred_imgs: n, w, h, 3
    # gt_imgs: n, w, h, 3
    # all in pixel values: 0 ~ 255
    # return: list of scores 0 ~ 1.
    assert len(pred_imgs) == len(gt_imgs)
    assert np.min(pred_imgs) >= 0 and np.min(gt_imgs) >= 0
    assert isinstance(metric, fid_wrapper) == False, 'FID not supported'
    corrects = []
    for idx, pred in enumerate(pred_imgs):
        gt = gt_imgs[idx]
        gt_score = metric(pred, gt)
        rest = [img for i, img in enumerate(gt_imgs) if i != idx] #all other gts
        count = 0
        for comp in rest:
            comp_score = metric(pred, comp)
            if is_sucess(gt_score, comp_score): #checks if gt_score is either > or < than comp_score (depending on metric)
                count += 1
        corrects.append(count / len(rest)) #for every predicted image, append ratio of how often the metric with its ground truth is superior compared to metric with all other images
    return corrects


def n_way_scores(pred_imgs, gt_imgs, metric, is_sucess, n=2, n_trials=100):
    # pred_imgs: n, w, h, 3
    # gt_imgs: n, w, h, 3
    # all in pixel values: 0 ~ 255
    # return: list of scores 0 ~ 1.
    assert len(pred_imgs) == len(gt_imgs)
    assert n <= len(pred_imgs) and n >= 2
    assert np.min(pred_imgs) >= 0 and np.min(gt_imgs) >= 0
    assert isinstance(metric, fid_wrapper) == False, 'FID not supported'
    corrects = []
    for idx, pred in enumerate(pred_imgs):
        gt = gt_imgs[idx]
        gt_score = metric(pred, gt)
        rest = np.stack([img for i, img in enumerate(gt_imgs) if i != idx])
        correct_count = 0
        for _ in range(n_trials):
            n_imgs_idx = np.random.choice(len(rest), n-1, replace=False)
            n_imgs = rest[n_imgs_idx]
            count = 0
            for comp in n_imgs:
                comp_score = metric(pred, comp)
                if is_sucess(gt_score, comp_score):
                    count += 1
            if count == len(n_imgs):
                correct_count += 1 
        corrects.append(correct_count / n_trials)
    return corrects


def metrics_only(pred_imgs, gt_imgs, metric, *args, **kwargs):
    assert np.min(pred_imgs) >= 0 and np.min(gt_imgs) >= 0
    return metric(pred_imgs, gt_imgs)


@torch.no_grad()
def n_way_top_k_acc(pred, class_id, n_way, num_trials=40, top_k=1):
    pick_range =[i for i in np.arange(len(pred)) if i != class_id]
    acc_list = []
    for t in range(num_trials):
        idxs_picked = np.random.choice(pick_range, n_way-1, replace=False)
        pred_picked = torch.cat([pred[class_id].unsqueeze(0), pred[idxs_picked]])
        acc = accuracy(
            pred_picked.unsqueeze(0), 
            torch.tensor([0], device=pred.device),
            num_classes = 50, 
            task = "multiclass", 
            top_k=top_k)
        acc_list.append(acc.item())
    return np.mean(acc_list), np.std(acc_list)


@torch.no_grad()
def get_n_way_top_k_acc(pred_imgs, ground_truth, n_way, num_trials, top_k, device, return_std=False):
    weights = ViT_H_14_Weights.DEFAULT
    model = vit_h_14(weights=weights)
    preprocess = weights.transforms()
    model = model.to(device)
    model = model.eval()
    
    acc_list = []
    std_list = []
    for pred, gt in zip(pred_imgs, ground_truth):
        pred = preprocess(Image.fromarray(pred.astype(np.uint8))).unsqueeze(0).to(device)
        gt = preprocess(Image.fromarray(gt.astype(np.uint8))).unsqueeze(0).to(device)
        gt_class_id = model(gt).squeeze(0).softmax(0).argmax().item()
        pred_out = model(pred).squeeze(0).softmax(0).detach()

        acc, std = n_way_top_k_acc(pred_out, gt_class_id, n_way, num_trials, top_k)
        acc_list.append(acc)
        std_list.append(std)
       
    if return_std:
        return acc_list, std_list
    return acc_list


def get_similarity_metric(img1, img2, method='pair-wise', metric_name='mse', **kwargs):
    # img1: n, w, h, 3
    # img2: n, w, h, 3
    # all in pixel values: 0 ~ 255
    # return: list of scores 0 ~ 1.
    if img1.shape[-1] != 3:
        img1 = rearrange(img1, 'n c w h -> n w h c')
    if img2.shape[-1] != 3:
        img2 = rearrange(img2, 'n c w h -> n w h c')

    if method == 'pair-wise':
        eval_procedure_func = pair_wise_score 
    elif method == 'n-way':
        eval_procedure_func = n_way_scores
    elif method == 'metrics-only':
        eval_procedure_func = metrics_only
    elif method == 'class':
        return get_n_way_top_k_acc(img1, img2, **kwargs)
    else:
        raise NotImplementedError

    if metric_name == 'mse':
        metric_func = mse_metric
        decision_func = smaller_the_better
    elif metric_name == 'pcc':
        metric_func = pcc_metric
        decision_func = larger_the_better
    elif metric_name == 'ssim':
        metric_func = ssim_metric
        decision_func = larger_the_better
    elif metric_name == 'psm':
        metric_func = psm_wrapper()
        decision_func = smaller_the_better
    elif metric_name == 'fid':
        metric_func = fid_wrapper()
        decision_func = smaller_the_better
    else:
        raise NotImplementedError
    
    return eval_procedure_func(img1, img2, metric_func, decision_func, **kwargs)

class EEGDataset():
    """
    EEG: the (preprocessed) EEG files are stored as .npy files (e.g.: ./data/preprocessed/wet/P00x/run_id/data.npy)
    with the individual image labels in the same directory (.../labels.npy) going from 0-599.
    The image labels can be mapped to the image paths via the experiment file used to run Psychopy which is found at
    ./psychopy/loopTemplate1.xlsx

    Images: the images for each image_class are stored in the directory ./data/images/experiment_subset_easy

    Args:
        data_dir: Path to directory containing the data.
            Expects a directory with multiple run-directories. Concatenates all npy files into one dataset.
        image_transform: Optional transform to be applied on images.
        train: Whether to use the training or validation set.
            if True, n-1 runs are returned as dataset
            if False, only the hold-out set is loaded for validation.
        val_run: Name of the run to be used as hold-out set for validation.
            Note: this allows to use a validation set from the same directory or from a different directory by specifying
            a new data_dir for the validation set.
        preload_images: Whether to pre-load all images into memory (speed vs memory trade-off)
    """
    def __init__(
            self, 
            data_dir: Path, 
            image_transform: Callable = None,
            train: bool = True, 
            val_run: str = None,
            preload_images: bool = False,
            ):
        self.preload_images = preload_images
        self.image_paths = np.array(pd.read_excel('./psychopy/loopTemplate1.xlsx', engine='openpyxl').images.values)
        self.image_transform = image_transform
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if all(p.is_dir() for p in Path(data_dir).iterdir()): #data_dir is a directory with multiple run-directories
            if train:
                self.labels = []
                self.data = []
                for run_dir in data_dir.iterdir():
                    if run_dir.name != val_run:
                        self.labels.append(np.load(run_dir / "labels.npy", allow_pickle=True))
                        self.data.append(np.load(run_dir / "data.npy", allow_pickle=True))
                self.labels = np.concatenate(self.labels)
                self.data = np.concatenate(self.data)
            else:
                #NOTE: REPLACE THIS WHEN EVERYTHING RUNS [::10]
                self.labels = np.load(data_dir / Path(val_run) / Path("labels.npy"), allow_pickle=True)[::10]
                self.data = np.load(data_dir / Path(val_run) / Path("data.npy"), allow_pickle=True)[::10]
        else:
            raise ValueError("data_dir should only contain run-directories")

        self.data = torch.from_numpy(self.data.swapaxes(1,2)).to(self.device) #swap axes to get (n_trials, channels, samples) 
        self.labels = torch.from_numpy(self.labels).long().to(self.device) #turn into one-hot encoding torch.nn.functional.one_hot( , num_classes = -1)
        
        if self.preload_images:
            self.images = np.stack([self.image_transform(np.array(Image.open(path[1:]).convert("RGB"))/255.0) for path in self.image_paths])
            self.images = torch.from_numpy(self.images).to(self.device)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.preload_images:
            img_idx = self.labels[idx]
            return {'eeg': self.data[idx].float(), 'image': self.images[img_idx]}
        else:
            image_raw = Image.open(self.image_paths[self.labels[idx]][1:]).convert("RGB")
            image = np.array(image_raw) / 255.0
            return {'eeg': self.data[idx].float(), 'image': self.image_transform(image)}


class Config_Generative_Model:
    def __init__(self):
        # project parameters
        self.seed = 2022
        self.root_path = '.'
        self.roi = 'VC'
        #self.patch_size = 16

        # self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains/ldm/semantic')
        self.pretrain_gm_path = os.path.join(self.root_path, 'reconstruction/pretrains/ldm/label2img') 
        # self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains/ldm/text2img-large')
        # self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains/ldm/layout2img')
        
        self.dataset = 'EEG'
        #self.pretrain_mbm_path = os.path.join(self.root_path, f'reconstruction/pretrains/{self.dataset}/fmri_encoder.pth') 

        self.img_size = 256

        np.random.seed(self.seed)
        # finetune parameters
        self.batch_size = 5 
        self.lr = 5.3e-5
        self.num_epoch = 200
        
        self.precision = 32
        self.accumulate_grad = 1
        self.crop_ratio = 0.2
        self.global_pool = False
        self.use_time_cond = True
        self.eval_avg = True

        # diffusion sampling parameters
        self.num_samples = 5
        self.ddim_steps = 250
        self.HW = None
        # resume check util
        self.model_meta = None
        self.checkpoint_path = None # os.path.join(self.root_path, 'results/generation/25-08-2022-08:02:55/checkpoint.pth')
        
        #Modifications
        self.eeg_data_path_train = "./data/preprocessed/wet/P001"
        self.eeg_data_path_test = "./data/test_sets/sub_P001/wet"
        self.eeg_val_path = "sub-P001_ses-S003_task-Default_run-003_eeg"
        self.eeg_model_config_path = "./reconstruction/pretrains/EEG/P001_model_config.yaml"
        self.eeg_model_ckpt_path = "./reconstruction/pretrains/EEG/final-model-P001.ckpt"


def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img

def channel_last(img):
        if img.shape[-1] == 3:
            return img
        return rearrange(img, 'c h w -> h w c')


class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Double Conditioning LDM Finetuning', add_help=False)
    # project parameters
    parser.add_argument('--seed', type=int)
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--pretrain_mbm_path', type=str)
    parser.add_argument('--crop_ratio', type=float)
    parser.add_argument('--dataset', type=str)

    # finetune parameters
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--precision', type=int)
    parser.add_argument('--accumulate_grad', type=int)
    parser.add_argument('--global_pool', type=bool)

    # diffusion sampling parameters
    parser.add_argument('--pretrain_gm_path', type=str)
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--ddim_steps', type=int)
    parser.add_argument('--use_time_cond', type=bool)
    parser.add_argument('--eval_avg', type=bool)
    
    args = parser.parse_args()
    config = Config_Generative_Model()

    for attr in config.__dict__:
        if hasattr(args, attr):
            if getattr(args, attr) != None:
                setattr(config, attr, getattr(args, attr))
    
    if config.checkpoint_path is not None:
        model_meta = torch.load(config.checkpoint_path, map_location='cpu')
        ckp = config.checkpoint_path
        config = model_meta['config']
        config.checkpoint_path = ckp
        print('Resuming from checkpoint: {}'.format(config.checkpoint_path))

    output_path = os.path.join(config.root_path, 'results', 'generation',  '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    config.output_path = output_path
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize Wandb
    wandb.init(project='EEG_reconstruction', group="stageB_dc-ldm", anonymous="allow", config=config, reinit=True)
    print(config.__dict__)
    with open(os.path.join(output_path, 'README.md'), 'w+') as f:
        print(config.__dict__, file=f)

    logger = WandbLogger()
    config.logger = logger
    
    # project setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    crop_pix = int(config.crop_ratio*config.img_size)
    img_transform_train = transforms.Compose([ normalize, transforms.Resize((256, 256)),  random_crop(config.img_size-crop_pix, p=0.5), transforms.Resize((256, 256)),  channel_last ])
    img_transform_test = transforms.Compose([ normalize,  transforms.Resize((256, 256)),  channel_last ])

    if config.dataset == "EEG":
        #subjects are selected by giving eeg_data_path
        eeg_latents_dataset_train = EEGDataset(data_dir = Path(config.eeg_data_path_train), val_run = None, train = True,image_transform = img_transform_train,preload_images = False) #set True if you have enough memory for speed up
        eeg_latents_dataset_test = EEGDataset(data_dir = Path(config.eeg_data_path_test), val_run = config.eeg_val_path, train = False, image_transform = img_transform_test, preload_images = False)
    else:
        raise NotImplementedError
    
    # create generative model (gm_path holds pretrained LDM)
    generative_model = eLDM(config.eeg_model_config_path, config.eeg_model_ckpt_path, device=device, pretrain_root=config.pretrain_gm_path, logger=config.logger, ddim_steps=config.ddim_steps, global_pool=config.global_pool, use_time_cond=config.use_time_cond)
    
    # resume training if applicable
    if config.checkpoint_path is not None:
        model_meta = torch.load(config.checkpoint_path, map_location='cpu')
        generative_model.model.load_state_dict(model_meta['model_state_dict'])
        print('model resumed')
        
    # finetune the model (use fast_dev_run=True for debugging)
    acc = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(accelerator=acc, max_epochs=config.num_epoch, logger=logger, precision=config.precision, accumulate_grad_batches=config.accumulate_grad_batches, enable_checkpointing=False, enable_model_summary=True, gradient_clip_val=0.5, check_val_every_n_epoch=5, fast_dev_run=False)
    generative_model.finetune(trainer, eeg_latents_dataset_train, eeg_latents_dataset_test, config.batch_size, config.lr, config.output_path, config=config)

    # Generate Train and Test Images after finetuning
    grid, _ = generative_model.generate(eeg_latents_dataset_train, config.num_samples, config.ddim_steps, config.HW, 10) # generate 10 instances
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    grid_imgs.save(os.path.join(config.output_path, 'samples_train.png'))
    wandb.log({'summary/samples_train': wandb.Image(grid_imgs)})

    grid, samples = generative_model.generate(eeg_latents_dataset_test, config.num_samples, config.ddim_steps, config.HW)
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    grid_imgs.save(os.path.join(config.output_path,f'./samples_test.png'))
    for sp_idx, imgs in enumerate(samples):
        for copy_idx, img in enumerate(imgs[1:]):
            img = rearrange(img, 'c h w -> h w c')
            Image.fromarray(img).save(os.path.join(config.output_path, f'./test{sp_idx}-{copy_idx}.png'))

    wandb.log({f'summary/samples_test': wandb.Image(grid_imgs)})

    # Get Metrics and Metric List to evaluate the predicted images compared to the ground truths
    metric_list = ['mse', 'pcc', 'ssim', 'psm']
    metric = []
    gt_images = [img[0] for img in samples]
    gt_images = rearrange(np.stack(gt_images), 'n c h w -> n h w c')
    samples_to_run = np.arange(1, len(samples[0])) if config.eval_avg else [1]
    for m in metric_list:
        res_part = []
        for s in samples_to_run:
            pred_images = [img[s] for img in samples]
            pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
            res = get_similarity_metric(pred_images, gt_images, method='pair-wise', metric_name=m)
            res_part.append(np.mean(res))
        metric.append(np.mean(res_part))     
    res_part = []
    for s in samples_to_run:
        pred_images = [img[s] for img in samples]
        pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
        res = get_similarity_metric(pred_images, gt_images, 'class', None, n_way=50, num_trials=50, top_k=1, device='cuda')
        res_part.append(np.mean(res))
    metric.append(np.mean(res_part))
    metric.append(np.max(res_part))
    metric_list.append('top-1-class')
    metric_list.append('top-1-class (max)')

    metric_dict = {f'summary/pair-wise_{k}':v for k, v in zip(metric_list[:-2], metric[:-2])}
    metric_dict[f'summary/{metric_list[-2]}'] = metric[-2]
    metric_dict[f'summary/{metric_list[-1]}'] = metric[-1]
    wandb.log(metric_dict)

    wandb.finish()