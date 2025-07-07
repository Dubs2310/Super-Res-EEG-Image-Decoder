import os
import sys
import torch
import numpy as np
import torch.nn as nn
from mne.channels import make_standard_montage, get_builtin_montages

def roundUpToNearestMultipleOfN(x, n):
    return int(x + (n - x % n) if x % n != 0 else x)


def generate_1d_positional_encoding(time_steps, d_s):
    pos_encoding = np.zeros((time_steps, d_s))
    for pos in range(time_steps):
        for i in range(d_s // 2):
            div_term = np.power(10000.0, (2 * i) / d_s)
            pos_encoding[pos, 2 * i] = np.sin(pos / div_term)
            pos_encoding[pos, 2 * i + 1] = np.cos(pos / div_term)
    pos_encoding = np.expand_dims(pos_encoding, axis=0)
    return torch.tensor(pos_encoding, dtype=torch.float32)


def generate_3d_positional_encoding(channel_names, d_t, builtin_montage=None, positions=[]):
    num_channels = len(channel_names)
    builtin_montages = get_builtin_montages()
    if num_channels == 0:
        raise ValueError("The number of channels must be greater than 0.")
    if builtin_montage and positions:
        raise ValueError("You can only use either builtin_montage or positions, not both.")
    if not builtin_montage and positions and len(positions) != num_channels:
        raise ValueError("The number of positions must match the number of channels.")
    if builtin_montage and not positions and builtin_montage not in builtin_montages:
        raise ValueError(f"Montage '{builtin_montage}' is not available. Please choose from {builtin_montages}.")
    
    ds_per_axis = d_t // 3
    if not builtin_montage and positions and len(positions) == num_channels:
        positions = np.array(positions)
    if builtin_montage and not positions and builtin_montage in builtin_montages:
        builtin_montage = make_standard_montage(builtin_montage)
        pos_dict = builtin_montage.get_positions()['ch_pos']
        positions = np.array([pos_dict[ch] for ch in channel_names])
    pos_encoding = []
    for axis in range(3):
        pos = positions[:, axis]
        pe = np.zeros((num_channels, ds_per_axis))
        for i in range(ds_per_axis):
            div_term = np.power(10000.0, (2 * i) / ds_per_axis)
            pe[:, i] = np.where(i % 2 == 0, np.sin(pos / div_term), np.cos(pos / div_term))
        pos_encoding.append(pe)
    pos_encoding = np.concatenate(pos_encoding, axis=-1)
    pos_encoding = np.expand_dims(pos_encoding, axis=0)
    return torch.tensor(pos_encoding, dtype=torch.float32)


class MaskTokenExpander(nn.Module):
    def __init__(self, mask_token):
        super().__init__()
        self.mask_token = nn.Parameter(mask_token)
    
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        return self.mask_token.repeat(batch_size, 1, 1)


class MaskTokensInsert(nn.Module):
    def __init__(self, lo_res_channel_names, hi_res_channel_names, mask_token):
        super().__init__()
        self.lr_dict = {ch: i for i, ch in enumerate(lo_res_channel_names)}
        self.hi_res_channel_names = hi_res_channel_names
        self.mask_expander = MaskTokenExpander(mask_token)
    
    def forward(self, inp):
        out_list = []
        for ch in self.hi_res_channel_names:
            if ch in self.lr_dict:
                idx = self.lr_dict[ch]
                ch_tensor = inp[:, idx:idx+1, :]
                out_list.append(ch_tensor)
            else:
                batch_reference = inp[:, 0:1, :]
                expanded_mask = self.mask_expander(batch_reference)
                out_list.append(expanded_mask)
        
        out = torch.cat(out_list, dim=1)
        return out


class SAB(nn.Module):
    def __init__(self, num_channels, time_steps, r_mlp, spatial_or_temporal="spatial", dropout_rate=0.5, L_t=1):
        super().__init__()
        self.is_temporal = spatial_or_temporal == "temporal"
        d_embed_in = time_steps if spatial_or_temporal == "spatial" else num_channels
        d_embed_out = d_embed_in * r_mlp
        num_attn_heads = 1 if self.is_temporal else 3
        
        self.layers = nn.ModuleList()
        for _ in range(L_t):
            layer = nn.ModuleDict({
                'norm1': nn.LayerNorm(d_embed_in, eps=1e-6),
                'attn': nn.MultiheadAttention(embed_dim=d_embed_in, num_heads=num_attn_heads, dropout=dropout_rate, batch_first=True),
                'dropout1': nn.Dropout(dropout_rate),
                'norm2': nn.LayerNorm(d_embed_in, eps=1e-6),
                'mlp': nn.Sequential(
                    nn.Linear(d_embed_in, d_embed_out),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(d_embed_out, d_embed_in),
                    nn.Dropout(dropout_rate)
                )
            })
            self.layers.append(layer)

    def forward(self, x):
        if self.is_temporal:
            x = x.permute(0, 2, 1)
        
        out = x
        for layer in self.layers:
            x_norm = layer['norm1'](out)
            attn_output, _ = layer['attn'](x_norm, x_norm, x_norm)
            out1 = out + layer['dropout1'](attn_output)
            out1_norm = layer['norm2'](out1)
            mlp_output = layer['mlp'](out1_norm)
            out2 = out1 + mlp_output
            out = out + out2
        
        if self.is_temporal:
            out = out.permute(0, 2, 1)
        
        return out


class CAB(nn.Module):
    def __init__(self, num_channels, time_steps, r_mlp, dropout_rate=0.5, L_s=1):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(L_s):
            tsab1 = SAB(num_channels=num_channels, time_steps=time_steps, r_mlp=r_mlp, spatial_or_temporal="temporal", dropout_rate=dropout_rate)
            ssab = SAB(num_channels=num_channels, time_steps=time_steps, r_mlp=r_mlp, spatial_or_temporal="spatial", dropout_rate=dropout_rate)
            tsab2 = SAB(num_channels=num_channels, time_steps=time_steps, r_mlp=r_mlp, spatial_or_temporal="temporal", dropout_rate=dropout_rate)
            block = nn.ModuleDict({ 'tsab1': tsab1, 'ssab': ssab, 'tsab2': tsab2})
            self.blocks.append(block)
    
    def forward(self, x):
        out = x
        for block in self.blocks:
            out1 = block['tsab1'](out)
            out1 = out + out1
            out2 = block['ssab'](out1)
            out2 = out1 + out2
            out3 = block['tsab2'](out2)
            out3 = out2 + out3
            out = out + out3
        return out


class SIM(nn.Module):
    def __init__(self, lo_res_channel_names, hi_res_channel_names, mask_token, time_steps, d_t, r_mlp, dropout_rate=0.5, builtin_montage=None, positions=[], L_s=1):
        super().__init__()
        self.low_res_ch_count = len(lo_res_channel_names)
        self.high_res_ch_count = len(hi_res_channel_names)

        low_res_3d_pos_encoding = generate_3d_positional_encoding(channel_names=lo_res_channel_names, d_t=d_t, builtin_montage=builtin_montage, positions=positions)
        high_res_3d_pos_encoding = generate_3d_positional_encoding(channel_names=hi_res_channel_names, d_t=d_t, builtin_montage=builtin_montage, positions=positions)
        self.register_buffer('low_res_3d_pos_encoding', low_res_3d_pos_encoding)
        self.register_buffer('high_res_3d_pos_encoding', high_res_3d_pos_encoding)

        self.dense1 = nn.Linear(time_steps, d_t)
        self.norm1 = nn.LayerNorm(d_t, eps=1e-6)
        self.cab1 = CAB(num_channels=self.low_res_ch_count, time_steps=d_t, r_mlp=r_mlp, dropout_rate=dropout_rate, L_s=L_s)
        self.mask_token_insert = MaskTokensInsert(lo_res_channel_names, hi_res_channel_names, mask_token)
        self.dense2 = nn.Linear(d_t, d_t)
        self.cab2 = CAB(num_channels=self.high_res_ch_count, time_steps=d_t, r_mlp=r_mlp, dropout_rate=dropout_rate, L_s=L_s)
        self.norm2 = nn.LayerNorm(d_t, eps=1e-6)
        self.dense3 = nn.Linear(d_t, time_steps)
        
    def forward(self, x):
        out = x
        out = self.dense1(out)
        out = out + self.low_res_3d_pos_encoding
        out = self.cab1(out)
        out = self.norm1(out)
        out = self.mask_token_insert(out)
        out = self.dense2(out)
        out = out + self.high_res_3d_pos_encoding
        out = self.cab2(out)
        out = self.norm2(out)
        out = self.dense3(out)
        return out


class TRM(nn.Module):
    def __init__(self, num_channels, time_steps, d_s, r_mlp, dropout_rate=0.5, L_t=1):
        super().__init__()
        self.register_buffer('_1d_pos_encoding', generate_1d_positional_encoding(time_steps, d_s))

        self.dense1 = nn.Linear(num_channels, d_s)
        self.tsab1 = SAB(num_channels=d_s, time_steps=time_steps, r_mlp=r_mlp, spatial_or_temporal="temporal", dropout_rate=dropout_rate, L_t=L_t)
        self.norm1 = nn.LayerNorm(time_steps, eps=1e-6)

        self.dense2 = nn.Linear(d_s, d_s)
        self.tsab2 = SAB(num_channels=d_s, time_steps=time_steps, r_mlp=r_mlp, spatial_or_temporal="temporal", dropout_rate=dropout_rate, L_t=L_t)
        self.norm2 = nn.LayerNorm(time_steps, eps=1e-6)
        
        self.dense3 = nn.Linear(d_s, num_channels)
        
    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = self.dense1(out)
        out = out + self._1d_pos_encoding
        out = out.permute(0, 2, 1)
        out = self.tsab1(out)
        out = self.norm1(out)
        out = out.permute(0, 2, 1)
        out = self.dense2(out)
        out = out + self._1d_pos_encoding
        out = out.permute(0, 2, 1)
        out = self.tsab2(out)
        out = self.norm2(out)
        out = out.permute(0, 2, 1)
        out = self.dense3(out)
        out = out.permute(0, 2, 1)
        return out
    

class SigmaParameters(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigma1 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.sigma2 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))


class ESTFormer(nn.Module):
    def __init__(self, lo_res_channel_names, hi_res_channel_names, time_steps, builtin_montage='standard_1020', alpha_t=0.60, alpha_s=0.75, r_mlp=4, dropout_rate=0.5, L_s=1, L_t=1):
        super().__init__()
        self.alpha_t = alpha_t
        self.alpha_s = alpha_s
        self.r_mlp = r_mlp
        self.L_s = L_s
        self.L_t = L_t
        self.num_lr_channels = len(lo_res_channel_names)
        self.num_hr_channels = len(hi_res_channel_names)
        self.time_steps = time_steps
        self.dropout_rate = dropout_rate

        self.sigmas = SigmaParameters()
        d_t = roundUpToNearestMultipleOfN(alpha_t * time_steps, 3)
        d_s = roundUpToNearestMultipleOfN(alpha_s * self.num_hr_channels, 2)
        self.mask_token = nn.Parameter(torch.zeros((1, 1, d_t)))

        self.sim = SIM(lo_res_channel_names=lo_res_channel_names, hi_res_channel_names=hi_res_channel_names, mask_token=self.mask_token, time_steps=time_steps, d_t=d_t, r_mlp=r_mlp, dropout_rate=dropout_rate, L_s=L_s, builtin_montage=builtin_montage)
        self.trm = TRM(num_channels=self.num_hr_channels, time_steps=time_steps, d_s=d_s, r_mlp=r_mlp, dropout_rate=dropout_rate, L_t=L_t)
        self.norm = nn.LayerNorm([self.num_hr_channels, time_steps], eps=1e-6)

    def forward(self, x):
        sim_out = self.sim(x)
        trm_out = self.trm(sim_out)
        out = sim_out + trm_out
        out = self.norm(out)
        return out

    def predict(self, eeg_lr):
        self.eval()
        with torch.no_grad():
            eeg_sr = self(eeg_lr)
            return eeg_sr