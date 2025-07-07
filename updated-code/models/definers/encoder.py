import torch
import numpy as np
from torch import nn
from open_clip.loss import ClipLoss
from braindecode.models import EEGNetv4
from models.definers.base import BaseModelDefiner

class EEGEncoderDefiner(BaseModelDefiner):
    def __init__(self, num_channels, timesteps):
        self.num_channels = num_channels
        self.timesteps = timesteps
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def register_model(self):
        return EEGNetv4(
            in_chans=self.num_channels, 
            input_window_samples=self.timesteps, 
            n_classes=1024, 
            final_conv_length='auto', 
            pool_mode='mean', 
            F1=8, 
            D=20,
            F2=160,
            kernel_length=4, 
            third_kernel_size=(4, 2), 
            drop_prob=0.25
        )
    
    def compute_loss(self, predictions, targets):
        return self.loss_func(predictions, targets, self.logit_scale)