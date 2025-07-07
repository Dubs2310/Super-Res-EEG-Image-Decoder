import torch
from models.definers.base import BaseModelDefiner
from models.core.super_resolution.estformer import ESTFormer

def loss(y_true, y_pred, sigma1, sigma2):
    fft_true = torch.fft.rfft(y_true.to(torch.float32))
    fft_pred = torch.fft.rfft(y_pred.to(torch.float32))
    fmse = torch.mean(torch.abs(fft_true - fft_pred) ** 2)
    lmae = torch.mean(torch.abs(y_true - y_pred))
    loss = (1.0 / (2.0 * sigma1**2)) * fmse + (1.0 / (2.0 * sigma2**2)) * lmae + torch.log(sigma1 * sigma2)
    return loss


class EEGSuperResolutionDefiner(BaseModelDefiner):
    def __init__(self, lo_res_channel_names, hi_res_channel_names, time_steps, builtin_montage='standard_1020', alpha_t=0.60, alpha_s=0.75, r_mlp=4, dropout_rate=0.5, L_s=1, L_t=1):
        self.lo_res_channel_names = lo_res_channel_names
        self.hi_res_channel_names = hi_res_channel_names
        self.time_steps = time_steps
        self.builtin_montage = builtin_montage
        self.alpha_t = alpha_t
        self.alpha_s = alpha_s
        self.r_mlp = r_mlp
        self.dropout_rate = dropout_rate
        self.L_s = L_s
        self.L_t = L_t
        super().__init__()

    def register_model(self):
        return ESTFormer(
            lo_res_channel_names=self.lo_res_channel_names,
            hi_res_channel_names=self.hi_res_channel_names,
            time_steps=self.time_steps,
            builtin_montage=self.builtin_montage,
            alpha_t=self.alpha_t,
            alpha_s=self.alpha_s,
            r_mlp=self.r_mlp,
            dropout_rate=self.dropout_rate,
            L_s=self.L_s,
            L_t=self.L_t
        )
    
    def compute_loss(self, predictions, targets):
        return loss(targets, predictions, self.model.sigmas.sigma1, self.model.sigmas.sigma2)