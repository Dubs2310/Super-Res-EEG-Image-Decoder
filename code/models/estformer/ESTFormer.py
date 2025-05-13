import os
import sys
import torch
import wandb
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from mne.channels import make_standard_montage, get_builtin_montages


sys.path.append(os.path.join(os.path.dirname(__file__), "..", '..'))
from utils.epoch_data_reader import EpochDataReader
from utils.metrics import (
    # mae as compute_mean_absolute_error, 
    # nmse as compute_normalized_mean_squared_error, 
    # pcc as compute_pearson_correlation_coefficient,
    # snr as compute_signal_to_noise_ratio,
    loss as reconstruction_loss,
)


def roundUpToNearestMultipleOfN(x, n):
    return int(x + (n - x % n) if x % n != 0 else x)


def generate_1d_positional_encoding(time_steps, d_s):
    """
    Args:
        time_steps: Total sequence length of a given EEG epoch
        d_s: Dimension length for the sin/cos time encoding, should be an even number
    Returns:
        Tensor with the shape (1, time_steps, d_s)
    """
    # assert d_s % 2 == 0, "d_s must be even for sin/cos encoding."
    pos_encoding = np.zeros((time_steps, d_s))
    for pos in range(time_steps):
        for i in range(d_s // 2):
            div_term = np.power(10000.0, (2 * i) / d_s)
            pos_encoding[pos, 2 * i] = np.sin(pos / div_term)
            pos_encoding[pos, 2 * i + 1] = np.cos(pos / div_term)
    pos_encoding = np.expand_dims(pos_encoding, axis=0)
    return torch.tensor(pos_encoding, dtype=torch.float32)


def generate_3d_positional_encoding(channel_names, d_t, builtin_montage=None, positions=[]):
    """
    Args:
        channel_names: List of names of EEG electrodes
        d_t: Dimension length for the 3D sin/cos channel encoding, should be divisible by 3
        builtin_montage: Name of the montage built into Python-MNE, shouldn't be used if positions is provided
        positions: 3D coordinates of the electrodes placed on the head in the order of channel names, shouldn't be used if builtin_montage is given
    Returns:
        Tensor with the shape (1, len(channel_names), d_t)
    """
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
    # assert d_t % 3 == 0, "d_t must be divisible by 3."
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
    """Custom module to properly handle mask token expansion for batch dimensions"""
    def __init__(self, mask_token):
        super().__init__()
        self.mask_token = nn.Parameter(mask_token)
    
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        return self.mask_token.repeat(batch_size, 1, 1)


class MaskTokensInsert(nn.Module):
    """
    Module to insert mask tokens at appropriate positions
    """
    def __init__(self, lr_channel_names, hr_channel_names, mask_token):
        """
        Args:
            lr_channel_names: list of low-res EEG channel names
            d_model: embedding dimension
            hr_channel_names: list of target high-res EEG channel names
            mask_token: Tensor of shape (1, 1, d_model), will be broadcast across batch
        """
        super().__init__()
        self.lr_dict = {ch: i for i, ch in enumerate(lr_channel_names)}
        self.hr_channel_names = hr_channel_names
        self.mask_expander = MaskTokenExpander(mask_token)
    
    def forward(self, inp):
        # inp shape: (batch_size, lr_channels, d_model)
        out_list = []
        
        for ch in self.hr_channel_names:
            if ch in self.lr_dict:
                idx = self.lr_dict[ch]
                ch_tensor = inp[:, idx:idx+1, :]  # Shape: (batch_size, 1, d_model)
                out_list.append(ch_tensor)
            else:
                # Use a reference tensor for batch dimension
                batch_reference = inp[:, 0:1, :]
                expanded_mask = self.mask_expander(batch_reference)
                out_list.append(expanded_mask)
        
        out = torch.cat(out_list, dim=1)  # Concatenate along channel dimension
        return out


class SAB(nn.Module):
    """Self Attention Block (Spatial/Temporal)"""
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
        # x shape: (batch_size, channels, time_steps)
        if self.is_temporal:
            x = x.permute(0, 2, 1)  # (batch_size, time_steps, channels)
        
        out = x
        for layer in self.layers:
            # First attention block
            x_norm = layer['norm1'](out)
            attn_output, _ = layer['attn'](x_norm, x_norm, x_norm)
            out1 = out + layer['dropout1'](attn_output)
            
            # MLP block
            out1_norm = layer['norm2'](out1)
            mlp_output = layer['mlp'](out1_norm)
            out2 = out1 + mlp_output
            
            # Residual connection
            out = out + out2
        
        if self.is_temporal:
            out = out.permute(0, 2, 1)  # (batch_size, channels, time_steps)
            
        return out


class CAB(nn.Module):
    """Cross Attention Block"""
    def __init__(self, num_channels, time_steps, r_mlp, dropout_rate=0.5, L_s=1):
        super().__init__()
        self.blocks = nn.ModuleList()
        
        for i in range(L_s):
            block = nn.ModuleDict({
                'tsab1': SAB(num_channels=num_channels, time_steps=time_steps, r_mlp=r_mlp, spatial_or_temporal="temporal", dropout_rate=dropout_rate),
                'ssab': SAB(num_channels=num_channels, time_steps=time_steps, r_mlp=r_mlp, spatial_or_temporal="spatial", dropout_rate=dropout_rate),
                'tsab2': SAB(num_channels=num_channels, time_steps=time_steps, r_mlp=r_mlp, spatial_or_temporal="temporal", dropout_rate=dropout_rate)
            })
            self.blocks.append(block)
    
    def forward(self, x):
        # x shape: (batch_size, num_channels, time_steps)
        out = x
        
        for block in self.blocks:
            # First temporal attention
            out1 = block['tsab1'](out)
            out1 = out + out1
            
            # Spatial attention
            out2 = block['ssab'](out1)
            out2 = out1 + out2
            
            # Second temporal attention
            out3 = block['tsab2'](out2)
            out3 = out2 + out3
            
            # Global residual connection
            out = out + out3
            
        return out


class SIM(nn.Module):
    """Spatial Interpolation Module"""
    def __init__(self, lr_channel_names, hr_channel_names, mask_token, time_steps, d_t, r_mlp, dropout_rate=0.5, builtin_montage=None, positions=[], L_s=1):
        super().__init__()
        self.low_res_ch_count = len(lr_channel_names)
        self.high_res_ch_count = len(hr_channel_names)
        
        # Generate positional encodings
        self.register_buffer('low_res_3d_pos_encoding', 
            generate_3d_positional_encoding(channel_names=lr_channel_names, d_t=d_t, builtin_montage=builtin_montage, positions=positions)
        )
        
        self.register_buffer('high_res_3d_pos_encoding', 
            generate_3d_positional_encoding(channel_names=hr_channel_names, d_t=d_t, builtin_montage=builtin_montage, positions=positions)
        )
        
        # Layers
        self.dense1 = nn.Linear(time_steps, d_t)
        self.norm1 = nn.LayerNorm(d_t, eps=1e-6)
        self.cab1 = CAB(num_channels=self.low_res_ch_count, time_steps=d_t, r_mlp=r_mlp, dropout_rate=dropout_rate, L_s=L_s)
        
        self.mask_token_insert = MaskTokensInsert(lr_channel_names, hr_channel_names, mask_token)
        
        self.dense2 = nn.Linear(d_t, d_t)
        self.cab2 = CAB(num_channels=self.high_res_ch_count, time_steps=d_t, r_mlp=r_mlp, dropout_rate=dropout_rate, L_s=L_s)
        self.norm2 = nn.LayerNorm(d_t, eps=1e-6)
        self.dense3 = nn.Linear(d_t, time_steps)
        
    def forward(self, x):
        # x shape: (batch_size, low_res_channels, time_steps)
        out = x
        
        # Apply first dense layer
        out = self.dense1(out)  # (batch_size, low_res_channels, d_t)
        
        # Add positional encoding
        out = out + self.low_res_3d_pos_encoding
        
        # Apply first CAB
        out = self.cab1(out)
        out = self.norm1(out)
        
        # Insert mask tokens
        out = self.mask_token_insert(out)  # (batch_size, high_res_channels, d_t)
        
        # Apply second dense layer
        out = self.dense2(out)
        
        # Add high-res positional encoding
        out = out + self.high_res_3d_pos_encoding
        
        # Apply second CAB
        out = self.cab2(out)
        out = self.norm2(out)
        
        # Apply final dense layer to restore original time dimension
        out = self.dense3(out)  # (batch_size, high_res_channels, time_steps)
        
        return out


class TRM(nn.Module):
    """Temporal Resolution Module"""
    def __init__(self, num_channels, time_steps, d_s, r_mlp, dropout_rate=0.5, L_t=1):
        super().__init__()
        
        # Register 1D positional encoding
        self.register_buffer('_1d_pos_encoding', generate_1d_positional_encoding(time_steps, d_s))
        
        # Layers
        self.dense1 = nn.Linear(num_channels, d_s)
        self.tsab1 = SAB(num_channels=d_s, time_steps=time_steps, r_mlp=r_mlp, spatial_or_temporal="temporal", dropout_rate=dropout_rate, L_t=L_t)
        self.norm1 = nn.LayerNorm(time_steps, eps=1e-6)
        
        self.dense2 = nn.Linear(d_s, d_s)
        self.tsab2 = SAB(num_channels=d_s, time_steps=time_steps, r_mlp=r_mlp, spatial_or_temporal="temporal", dropout_rate=dropout_rate, L_t=L_t)
        self.norm2 = nn.LayerNorm(time_steps, eps=1e-6)
        
        self.dense3 = nn.Linear(d_s, num_channels)
        
    def forward(self, x):
        # x shape: (batch_size, num_channels, time_steps)
        
        # Transpose to (batch_size, time_steps, num_channels)
        out = x.permute(0, 2, 1)
        
        # First dense layer and positional encoding
        out = self.dense1(out)
        out = out + self._1d_pos_encoding
        
        # Transpose back for first TSAB
        out = out.permute(0, 2, 1)  # (batch_size, d_s, time_steps)
        out = self.tsab1(out)
        out = self.norm1(out)
        
        # Transpose again for second TSAB
        out = out.permute(0, 2, 1)  # (batch_size, time_steps, d_s)
        out = self.dense2(out)
        out = out + self._1d_pos_encoding
        out = out.permute(0, 2, 1)  # (batch_size, d_s, time_steps)
        
        out = self.tsab2(out)
        out = self.norm2(out)
        
        # Final processing
        out = out.permute(0, 2, 1)  # (batch_size, time_steps, d_s)
        out = self.dense3(out)
        out = out.permute(0, 2, 1)  # (batch_size, num_channels, time_steps)
        
        return out
    

class SigmaParameters(nn.Module):
    """Class to hold trainable sigma parameters"""
    def __init__(self):
        super().__init__()
        self.sigma1 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.sigma2 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))


class ESTFormer(nn.Module):
    """ESTFormer Model for EEG Super-resolution"""
    def __init__(self, device, lr_channel_names, hr_channel_names, builtin_montage, time_steps, alpha_t, alpha_s, r_mlp, dropout_rate, L_s, L_t):
        super().__init__()
        self.device = device
        self.alpha_t = alpha_t
        self.alpha_s = alpha_s
        self.r_mlp = r_mlp
        self.L_s = L_s
        self.L_t = L_t
        self.num_lr_channels = len(lr_channel_names)
        self.num_hr_channels = len(hr_channel_names)
        self.time_steps = time_steps
        self.dropout_rate = dropout_rate

        self.sigmas = SigmaParameters()
        d_t = roundUpToNearestMultipleOfN(alpha_t * time_steps, 3)
        d_s = roundUpToNearestMultipleOfN(alpha_s * self.num_hr_channels, 2)
        self.mask_token = nn.Parameter(torch.zeros((1, 1, d_t)))
        
        # Create main modules
        self.sim = SIM(lr_channel_names=lr_channel_names, hr_channel_names=hr_channel_names, mask_token=self.mask_token, time_steps=time_steps, d_t=d_t, r_mlp=r_mlp, dropout_rate=dropout_rate, L_s=L_s, builtin_montage=builtin_montage)
        self.trm = TRM(num_channels=self.num_hr_channels, time_steps=time_steps, d_s=d_s, r_mlp=r_mlp, dropout_rate=dropout_rate, L_t=L_t)
        self.norm = nn.LayerNorm([self.num_hr_channels, time_steps], eps=1e-6)
        self.to(self.device)


    def forward(self, x):
        # x shape: (batch_size, lr_channels, time_steps)
        sim_out = self.sim(x)
        trm_out = self.trm(sim_out)
        out = sim_out + trm_out
        out = self.norm(out)
        return out
        
    def training_pass(self, epoch):
        self.train()
        self.sigmas.train()
        self.lo_res_loader.dataset.set_split_type('train')
        self.hi_res_loader.dataset.set_split_type('train')

        train_losses = []
        # train_maes = []
        # train_nmse = []
        # train_snr = []
        # train_pcc = []

        progress_bar = tqdm(zip(self.lo_res_loader, self.hi_res_loader), desc=f"Epoch {epoch+1}/{self.epochs}", leave=True, total=min(len(self.lo_res_loader), len(self.hi_res_loader)))

        for i, (lo_res_batch, hi_res_batch) in enumerate(progress_bar):
            if len(lo_res_batch) == 2 and len(hi_res_batch) == 2:
                lo_res = lo_res_batch[0].float().to(self.device)
                hi_res = hi_res_batch[0].float().to(self.device)
            else:
                lo_res = lo_res_batch.float().to(self.device)
                hi_res = hi_res_batch.float().to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self(lo_res)

            # Compute loss
            loss = reconstruction_loss(hi_res, outputs, self.sigmas.sigma1, self.sigmas.sigma2)
            # mae = compute_mean_absolute_error(hi_res, outputs)
            # nmse = compute_normalized_mean_squared_error(hi_res, outputs)
            # snr = compute_signal_to_noise_ratio(hi_res, outputs)
            # pcc = compute_pearson_correlation_coefficient(hi_res, outputs)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Track metrics
            train_losses.append(loss.item())
            # train_maes.append(mae.item())
            # train_nmse.append(nmse.item())
            # train_snr.append(snr.item())
            # train_pcc.append(pcc.item())

            # Update tqdm bar with metrics
            if i % 10 == 0:
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}"
                    # mae=f"{mae.item():.4f}", 
                    # nmse=f"{nmse.item():.4f}", 
                    # snr=f"{snr.item():.4f}", 
                    # pcc=f"{pcc.item():.4f}"
                )

            # Free memory
            # del lo_res, hi_res, outputs, loss#, mae, nmse, snr, pcc
            # torch.cuda.empty_cache()
        
        # Compute epoch metrics
        avg_train_loss = np.mean(train_losses)
        # avg_train_mae = np.mean(train_maes)
        # avg_train_nmse = np.mean(train_nmse)
        # avg_train_snr = np.mean(train_snr)
        # avg_train_pcc = np.mean(train_pcc)

        return {
           'loss': avg_train_loss, 
        #    'mae': avg_train_mae, 
        #    'nmse': avg_train_nmse, 
        #    'snr': avg_train_snr, 
        #    'pcc': avg_train_pcc
        }

    def validation_pass(self):
        self.eval()
        self.sigmas.eval()
        self.lo_res_loader.dataset.set_split_type('val')
        self.hi_res_loader.dataset.set_split_type('val')

        val_losses = []
        # val_maes = []
        # val_nmse = []
        # val_snr = []
        # val_pcc = []
        
        with torch.no_grad():
            progress_bar = tqdm(zip(self.lo_res_loader, self.hi_res_loader), desc=f"Running model on validation set...", leave=False, total=min(len(self.lo_res_loader), len(self.hi_res_loader)))
            for lo_res_batch, hi_res_batch in progress_bar:
                if len(lo_res_batch) == 2 and len(hi_res_batch) == 2:
                    lo_res = lo_res_batch[0].float().to(self.device)
                    hi_res = hi_res_batch[0].float().to(self.device)
                else:
                    lo_res = lo_res_batch.float().to(self.device)
                    hi_res = hi_res_batch.float().to(self.device)

                # Forward pass
                outputs = self(lo_res)
                
                # Compute loss
                loss = reconstruction_loss(hi_res, outputs, self.sigmas.sigma1, self.sigmas.sigma2)
                # mae = compute_mean_absolute_error(hi_res, outputs)
                # nmse = compute_normalized_mean_squared_error(hi_res, outputs)
                # snr = compute_signal_to_noise_ratio(hi_res, outputs)
                # pcc = compute_pearson_correlation_coefficient(hi_res, outputs)
                
                # Track metrics
                val_losses.append(loss.item())
                # val_maes.append(mae.item())
                # val_nmse.append(nmse.item())
                # val_snr.append(snr.item())
                # val_pcc.append(pcc.item())

                # Free memory
                # del lo_res, hi_res, outputs, loss#, mae, nmse, snr, pcc
                # torch.cuda.empty_cache()
        
        # Compute epoch metrics
        avg_val_loss = np.mean(val_losses)
        # avg_val_mae = np.mean(val_maes)
        # avg_val_nmse = np.mean(val_nmse)
        # avg_val_snr = np.mean(val_snr)
        # avg_val_pcc = np.mean(val_pcc)

        return {
           'loss': avg_val_loss, 
        #    'mae': avg_val_mae, 
        #    'nmse': avg_val_nmse, 
        #    'snr': avg_val_snr, 
        #    'pcc': avg_val_pcc
        }

    def fit(self, epochs, lo_res_loader, hi_res_loader, optimizer, checkpoint_dir, identifier):
        os.makedirs('checkpoints', exist_ok=True)
        self.epochs = epochs
        self.lo_res_loader = lo_res_loader
        self.hi_res_loader = hi_res_loader
        self.optimizer = optimizer

        metrics = ['loss']#, 'mae', 'nmse', 'pcc', 'snr']
        history = { 'sigma1': [], 'sigma2': [] }
        for m in metrics:
            history[f'train_{m}'] = []
            history[f'val_{m}'] = []

        self.best_val_loss = float('inf')

        for epoch in range(epochs):
            train_metrics = self.training_pass(epoch)
            val_metrics = self.validation_pass()
            
            log_object = { "epoch": epoch + 1 }
            summary_str = f"Epoch {epoch + 1}/{epochs}, "

            for k, v in train_metrics.items():
                key = f'train_{k}'
                log_object[key] = v
                history[key].append(v)
                summary_str += f'{key}: {v:.4f}'

            for k, v in val_metrics.items():
                key = f'val_{k}'
                log_object[key] = v
                history[key].append(v)
                summary_str += f'{key}: {v:.4f}'
            
            log_object['sigma1'] = (s1 := self.sigmas.sigma1.item())
            log_object['sigma2'] = (s2 := self.sigmas.sigma2.item())
            history['sigma1'].append(s1)
            history['sigma2'].append(s2)
            summary_str += (
                f"sigma1: {s1:.4f}, "
                f"sigma2: {s2:.4f}"
            )

            wandb.log(log_object)
            print(summary_str)

            avg_val_loss = log_object['val_loss']
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                checkpoint_path = os.path.join(checkpoint_dir, f'estformer_{identifier}_best.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'sigmas_state_dict': self.sigmas.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                }, checkpoint_path)
                print(f"Saved best model checkpoint to {checkpoint_path}")
                wandb.save(checkpoint_path, policy='now') # Log best model to wandb
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'estformer_{identifier}_epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'sigmas_state_dict': self.sigmas.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                }, checkpoint_path)
        
        return history
    
    def predict(self, eeg_lr):
        self.eval()
        with torch.no_grad():
            eeg_sr = self(eeg_lr)
            return eeg_sr
    
    # def predict_on_test_set(self, dataloader):
    #     self.eval()

    #     self.sigmas.eval()
    #     test_losses = []
    #     test_maes = []
    #     test_nmse = []
    #     test_snr = []
    #     test_pcc = []
        
    #     with torch.no_grad():
    #         progress_bar = tqdm(dataloader, desc=f"Running model on test set...", leave=True)
    #         for batch in progress_bar:
    #             lo_res = batch['lo_res'].float().to(self.device)
    #             hi_res = batch['hi_res'].float().to(self.device)
                
    #             # Forward pass
    #             outputs = self(lo_res)
                
    #             # Compute loss
    #             loss = reconstruction_loss(hi_res, outputs, self.sigmas.sigma1, self.sigmas.sigma2)
    #             mae = compute_mean_absolute_error(hi_res, outputs)
    #             nmse = compute_normalized_mean_squared_error(hi_res, outputs)
    #             snr = compute_signal_to_noise_ratio(hi_res, outputs)
    #             pcc = compute_pearson_correlation_coefficient(hi_res, outputs)
                
    #             # Track metrics
    #             test_losses.append(loss.item())
    #             test_maes.append(mae.item())
    #             test_nmse.append(nmse.item())
    #             test_snr.append(snr.item())
    #             test_pcc.append(pcc.item())

    #             # Free memory
    #             del lo_res, hi_res, outputs, loss, mae
    #             torch.cuda.empty_cache()
        
    #     # Compute epoch metrics
    #     avg_test_loss = np.mean(test_losses)
    #     avg_test_mae = np.mean(test_maes)
    #     avg_test_nmse = np.mean(test_nmse)
    #     avg_test_snr = np.mean(test_snr)
    #     avg_test_pcc = np.mean(test_pcc)

    #     avg_results = {
    #        'avg_test_loss': avg_test_loss, 
    #        'avg_test_mae': avg_test_mae, 
    #        'avg_test_nmse': avg_test_nmse, 
    #        'avg_test_snr': avg_test_snr, 
    #        'avg_test_pcc': avg_test_pcc
    #     }

    #     return avg_results
    
    # def upsample(self, x, save_to_h5=True):
        # self.



# def monitor_sigma_values_and_loss(history):
#     """
#     Monitor the values of sigma1 and sigma2 during training.
    
#     Args:
#         history: Training history dictionary
#     """
#     # Get the values of sigma1 and sigma2
#     sigma1_values = history['sigma1']
#     sigma2_values = history['sigma2']
    
#     print(f"Final sigma1 value: {sigma1_values[-1]}")
#     print(f"Final sigma2 value: {sigma2_values[-1]}")
    
#     # Plot the loss history
#     plt.figure(figsize=(12, 8))
    
#     # Plot loss
#     plt.subplot(2, 2, 1)
#     plt.plot(history['train_loss'], label='Training Loss')
#     plt.plot(history['val_loss'], label='Validation Loss')
#     plt.title('Model Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
    
#     # Plot MAE
#     plt.subplot(2, 2, 2)
#     plt.plot(history['train_mae'], label='Training MAE')
#     plt.plot(history['val_mae'], label='Validation MAE')
#     plt.title('Model MAE')
#     plt.xlabel('Epoch')
#     plt.ylabel('MAE')
#     plt.legend()

#     # Plot NMSE
#     plt.subplot(2, 2, 3)
#     plt.plot(history['train_nmse'], label='Training NMSE')
#     plt.plot(history['val_nmse'], label='Validation NMSE')
#     plt.title('Model NMSE')
#     plt.xlabel('Epoch')
#     plt.ylabel('NMSE')
#     plt.legend()

#     # Plot SNR
#     plt.subplot(2, 2, 4)
#     plt.plot(history['train_snr'], label='Training SNR')
#     plt.plot(history['val_snr'], label='Validation SNR')
#     plt.title('Model SNR')
#     plt.xlabel('Epoch')
#     plt.ylabel('SNR')
#     plt.legend()
    
#     # Plot PCC
#     plt.subplot(2, 2, 5)
#     plt.plot(history['train_pcc'], label='Training PCC')
#     plt.plot(history['val_pcc'], label='Validation PCC')
#     plt.title('Model PCC')
#     plt.xlabel('Epoch')
#     plt.ylabel('PCC')
#     plt.legend()
    
#     # Plot sigma values
#     plt.subplot(2, 2, 3)
#     plt.plot(sigma1_values, label='Sigma1')
#     plt.title('Sigma1 Value')
#     plt.xlabel('Epoch')
#     plt.ylabel('Value')
    
#     plt.subplot(2, 2, 4)
#     plt.plot(sigma2_values, label='Sigma2')
#     plt.title('Sigma2 Value')
#     plt.xlabel('Epoch')
#     plt.ylabel('Value')
    
#     plt.tight_layout()
    
#     # Save figure to wandb
#     if wandb.run is not None:
#         wandb.log({"training_history": wandb.Image(plt)})
    
#     plt.show()



# def main():
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # Force CUDA to use the GPU
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
#     os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # Enable memory optimization settings for PyTorch

#     # Check if CUDA is available
#     import GPUtil
#     try:
#         gpus = GPUtil.getGPUs()
#         if gpus:
#             print(f"GPUtil detected {len(gpus)} GPUs:")
#             for i, gpu in enumerate(gpus):
#                 print(f"  GPU {i}: {gpu.name} (Memory: {gpu.memoryTotal}MB)")
            
#             # Set default GPU
#             os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(len(gpus))])
#             print(f"Set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
#         else:
#             print("GPUtil found no available GPUs")
#     except Exception as e:
#         print(f"Error checking GPUs with GPUtil: {e}")

#     # Check for CUDA availability
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")

#     # Print available GPU memory
#     if torch.cuda.is_available():
#         print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
#         print(f"Available GPU memory: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

#     all_channels = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']

#     # Model parameters
#     hr_channel_names = all_channels # High-resolution setup (all channels)
#     lr_channel_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'] # Low-resolution setup (fewer channels)
#     builtin_montage = 'standard_1020'
#     alpha_t = 0.60
#     alpha_s = 0.75
#     r_mlp = 4 # amplification factor for MLP layers
#     dropout_rate = 0.5
#     L_s = 1  # Number of spatial layers
#     L_t = 1  # Number of temporal layers

#     # Training parameters
#     epochs = 30

#     # Optimizer parameters
#     lr = 5e-5
#     weight_decay = 0.5
#     beta_1 = 0.9
#     beta_2 = 0.95

#     # Dataset parameters
#     batch_size = 30
#     dataset_split = "70/25/5"
#     eeg_epoch_mode = "fixed_length"
#     fixed_length_duration = 6
#     duration_before_onset = 0.05
#     duration_after_onset = 0.6
#     random_state = 97
    
#     # Create datasets
#     train_dataset = HDF5DataSplitGenerator(
#         dataset_type="train",
#         dataset_split=dataset_split,
#         eeg_epoch_mode=eeg_epoch_mode,
#         random_state=random_state,
#         fixed_length_duration=fixed_length_duration,
#         duration_before_onset=duration_before_onset,
#         duration_after_onset=duration_after_onset,
#         lr_channel_names=lr_channel_names,
#         hr_channel_names=hr_channel_names
#     )
    
#     val_dataset = HDF5DataSplitGenerator(
#         dataset_type="val",
#         dataset_split=dataset_split,
#         eeg_epoch_mode=eeg_epoch_mode,
#         random_state=random_state,
#         fixed_length_duration=fixed_length_duration,
#         duration_before_onset=duration_before_onset,
#         duration_after_onset=duration_after_onset,
#         lr_channel_names=lr_channel_names,
#         hr_channel_names=hr_channel_names
#     )

#     test_dataset = HDF5DataSplitGenerator(
#         dataset_type="val",
#         dataset_split=dataset_split,
#         eeg_epoch_mode=eeg_epoch_mode,
#         random_state=random_state,
#         fixed_length_duration=fixed_length_duration,
#         duration_before_onset=duration_before_onset,
#         duration_after_onset=duration_after_onset,
#         lr_channel_names=lr_channel_names,
#         hr_channel_names=hr_channel_names
#     )
    
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True
#     )

#     test_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True
#     )
    
#     # Get sample data to determine time_steps
#     sample_item = train_loader.dataset[0]
#     time_steps = sample_item["lo_res"].shape[1]
#     sfreq = sample_item["sfreq"]

#     model = ESTFormer(
#         device, 
#         lr_channel_names=lr_channel_names,
#         hr_channel_names=hr_channel_names,
#         builtin_montage=builtin_montage,
#         time_steps=time_steps,
#         alpha_t=alpha_t,
#         alpha_s=alpha_s,
#         r_mlp=r_mlp,
#         dropout_rate=dropout_rate,
#         L_s=L_s,
#         L_t=L_t
#     )

#     config = {
#         "total_epochs_trained_on": epochs,
#         "scale_factor": len(hr_channel_names) / len(lr_channel_names),
#         "time_steps_in_seconds": time_steps / sfreq,
#         "is_parieto_occipital_exclusive": all(ch.startswith(('P', 'O')) for ch in lr_channel_names) and all(ch.startswith(('P', 'O')) for ch in hr_channel_names),
#         "model_params": {
#             "model": "ESTformer",
#             "num_lr_channels": len(lr_channel_names),
#             "num_hr_channels": len(hr_channel_names),
#             "builtin_montage": builtin_montage,
#             "alpha_s": alpha_s,
#             "alpha_t": alpha_t,
#             "r_mlp": r_mlp,
#             "dropout_rate": dropout_rate,
#             "L_s": L_s,
#             "L_t": L_t,
#         },
#         "dataset_params": {
#             "eeg_epoch_mode": eeg_epoch_mode,
#             "dataset_split": dataset_split,
#             "fixed_length_duration": fixed_length_duration,
#             "duration_before_onset": duration_before_onset,
#             "duration_after_onset": duration_after_onset,
#             "batch_size": batch_size,
#             "random_state": random_state
#         },
#         "optimizer_params": {
#             "optimizer": "Adam",
#             "learning_rate": lr,
#             "weight_decay": weight_decay,
#             "betas": (beta_1, beta_2)
#         }
#     }
    
#     wandb.init(project="eeg-estformer", config=config)

#     # Create optimizer with both model and sigma parameters
#     optimizer = optim.Adam(
#         params=[{'params': model.parameters()}], 
#         lr=lr,
#         weight_decay=weight_decay, 
#         betas=(beta_1, beta_2)
#     )

#     history = model.fit(
#         epochs=epochs,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         optimizer=optimizer,
#         checkpoint_dir='checkpoints'
#     )
#     
#     average_test_results = model.predict_on_test_set(test_loader)
#     print("Average Results on Test Set: ", average_test_results)
#
#     # monitor_sigma_values_and_loss(history)
#     # visualize_results(model, val_loader.dataset, device)
#     print("Training completed successfully!")

#     wandb.finish()

# if __name__ == '__main__':
#     main()