import torch
import torch.nn as nn
import numpy as np
from mne.channels import make_standard_montage, get_builtin_montages

def generate_1d_positional_encoding(time_steps, d_model):
    """
    Args:
        time_steps: Total sequence length of a given EEG epoch
        d_model: Dimension length for the sin/cos time encoding, should be an even number
    Returns:
        Tensor with the shape (1, time_steps, d_model)
    """
    assert d_model % 2 == 0, "d_model must be even for sin/cos encoding."
    pos_encoding = np.zeros((time_steps, d_model))
    for pos in range(time_steps):
        for i in range(d_model // 2):
            div_term = np.power(10000.0, (2 * i) / d_model)
            pos_encoding[pos, 2 * i] = np.sin(pos / div_term)
            pos_encoding[pos, 2 * i + 1] = np.cos(pos / div_term)
    pos_encoding = np.expand_dims(pos_encoding, axis=0)
    return torch.tensor(pos_encoding, dtype=torch.float32)

def generate_3d_positional_encoding(channel_names, d_model, builtin_montage=None, positions=[]):
    """
    Args:
        channel_names: List of names of EEG electrodes
        d_model: Dimension length for the 3D sin/cos channel encoding, should be divisible by 3
        builtin_montage: Name of the montage built into Python-MNE, shouldn't be used if positions is provided
        positions: 3D coordinates of the electrodes placed on the head in the order of channel names, shouldn't be used if builtin_montage is given
    Returns:
        Tensor with the shape (1, len(channel_names), d_model)
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
    assert d_model % 3 == 0, "d_model must be divisible by 3."
    d_model_per_axis = d_model // 3
    if not builtin_montage and positions and len(positions) == num_channels:
        positions = np.array(positions)
    if builtin_montage and not positions and builtin_montage in builtin_montages:
        builtin_montage = make_standard_montage(builtin_montage)
        pos_dict = builtin_montage.get_positions()['ch_pos']
        positions = np.array([pos_dict[ch] for ch in channel_names])
    pos_encoding = []
    for axis in range(3):
        pos = positions[:, axis]
        pe = np.zeros((num_channels, d_model_per_axis))
        for i in range(d_model_per_axis):
            div_term = np.power(10000.0, (2 * i) / d_model_per_axis)
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
    def __init__(self, lr_channel_names, d_model, hr_channel_names, mask_token):
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
        # batch_size = inp.shape[0]
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
    def __init__(self, num_channels, time_steps, num_heads, mlp_dim, spatial_or_temporal="spatial", dropout_rate=0.1, L=1):
        super().__init__()
        self.is_temporal = spatial_or_temporal == "temporal"
        embed_dim = time_steps if spatial_or_temporal == "spatial" else num_channels
        # print(embed_dim, num_heads)
        
        self.layers = nn.ModuleList()
        for _ in range(L):
            layer = nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim, eps=1e-6),
                'attn': nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True),
                'dropout1': nn.Dropout(dropout_rate),
                'norm2': nn.LayerNorm(embed_dim, eps=1e-6),
                'mlp': nn.Sequential(
                    nn.Linear(embed_dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(mlp_dim, embed_dim),
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
    def __init__(self, num_channels, time_steps, num_heads, mlp_dim, dropout_rate=0.1, L=1):
        super().__init__()
        self.blocks = nn.ModuleList()
        
        for i in range(L):
            block = nn.ModuleDict({
                'tsab1': SAB(num_channels=num_channels, time_steps=time_steps, num_heads=num_heads, mlp_dim=mlp_dim, spatial_or_temporal="temporal", dropout_rate=dropout_rate),
                'ssab': SAB(num_channels=num_channels, time_steps=time_steps, num_heads=num_heads, mlp_dim=mlp_dim, spatial_or_temporal="spatial", dropout_rate=dropout_rate),
                'tsab2': SAB(num_channels=num_channels, time_steps=time_steps, num_heads=num_heads, mlp_dim=mlp_dim, spatial_or_temporal="temporal", dropout_rate=dropout_rate)
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
    def __init__(self, lr_channel_names, hr_channel_names, mask_token, time_steps, d_model, num_heads, mlp_dim, dropout_rate=0.1, builtin_montage=None, positions=[], L=1):
        super().__init__()
        self.low_res_ch_count = len(lr_channel_names)
        self.high_res_ch_count = len(hr_channel_names)
        
        # Generate positional encodings
        self.register_buffer('low_res_3d_pos_encoding', 
            generate_3d_positional_encoding(channel_names=lr_channel_names, d_model=d_model, builtin_montage=builtin_montage, positions=positions)
        )
        
        self.register_buffer('high_res_3d_pos_encoding', 
            generate_3d_positional_encoding(channel_names=hr_channel_names, d_model=d_model, builtin_montage=builtin_montage, positions=positions)
        )
        
        # Layers
        self.dense1 = nn.Linear(time_steps, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.cab1 = CAB(num_channels=self.low_res_ch_count, time_steps=d_model, num_heads=num_heads, mlp_dim=mlp_dim, dropout_rate=dropout_rate, L=L)
        
        self.mask_token_insert = MaskTokensInsert(lr_channel_names, d_model, hr_channel_names, mask_token)
        
        self.dense2 = nn.Linear(d_model, d_model)
        self.cab2 = CAB(num_channels=self.high_res_ch_count, time_steps=d_model, num_heads=num_heads, mlp_dim=mlp_dim, dropout_rate=dropout_rate, L=L)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dense3 = nn.Linear(d_model, time_steps)
        
    def forward(self, x):
        # x shape: (batch_size, low_res_channels, time_steps)
        out = x
        
        # Apply first dense layer
        out = self.dense1(out)  # (batch_size, low_res_channels, d_model)
        
        # Add positional encoding
        out = out + self.low_res_3d_pos_encoding
        
        # Apply first CAB
        out = self.cab1(out)
        out = self.norm1(out)
        
        # Insert mask tokens
        out = self.mask_token_insert(out)  # (batch_size, high_res_channels, d_model)
        
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
    def __init__(self, num_channels, time_steps, d_model, num_heads, mlp_dim, dropout_rate=0.1, L=1):
        super().__init__()
        
        # Register 1D positional encoding
        self.register_buffer('_1d_pos_encoding', generate_1d_positional_encoding(time_steps, d_model))
        
        # Layers
        self.dense1 = nn.Linear(num_channels, d_model)
        self.tsab1 = SAB(num_channels=d_model, time_steps=time_steps, num_heads=num_heads, mlp_dim=mlp_dim, spatial_or_temporal="temporal", dropout_rate=dropout_rate, L=L)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        
        self.dense2 = nn.Linear(d_model, d_model)
        self.tsab2 = SAB(num_channels=d_model, time_steps=time_steps, num_heads=num_heads, mlp_dim=mlp_dim, spatial_or_temporal="temporal", dropout_rate=dropout_rate, L=L)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        self.dense3 = nn.Linear(d_model, num_channels)
        
    def forward(self, x):
        # x shape: (batch_size, num_channels, time_steps)
        
        # Transpose to (batch_size, time_steps, num_channels)
        out = x.permute(0, 2, 1)
        
        # First dense layer and positional encoding
        out = self.dense1(out)
        out = out + self._1d_pos_encoding
        
        # Transpose back for first TSAB
        out = out.permute(0, 2, 1)  # (batch_size, d_model, time_steps)
        out = self.tsab1(out)
        out = self.norm1(out)
        
        # Transpose again for second TSAB
        out = out.permute(0, 2, 1)  # (batch_size, time_steps, d_model)
        out = self.dense2(out)
        out = out + self._1d_pos_encoding
        out = out.permute(0, 2, 1)  # (batch_size, d_model, time_steps)
        
        out = self.tsab2(out)
        out = self.norm2(out)
        
        # Final processing
        out = out.permute(0, 2, 1)  # (batch_size, time_steps, d_model)
        out = self.dense3(out)
        out = out.permute(0, 2, 1)  # (batch_size, num_channels, time_steps)
        
        return out

class ESTFormer(nn.Module):
    """ESTFormer Model for EEG Super-resolution"""
    def __init__(self, lr_channel_names, hr_channel_names, builtin_montage, time_steps, d_model, num_heads, mlp_dim, dropout_rate, Ls, Lt):
        super().__init__()
        
        # Create mask token as a learnable parameter
        self.mask_token = nn.Parameter(torch.zeros((1, 1, d_model)))
        
        # Create main modules
        self.sim = SIM(lr_channel_names=lr_channel_names, hr_channel_names=hr_channel_names, mask_token=self.mask_token,time_steps=time_steps, d_model=d_model, num_heads=num_heads, mlp_dim=mlp_dim, dropout_rate=dropout_rate, L=Ls, builtin_montage=builtin_montage)
        self.trm = TRM(num_channels=len(hr_channel_names), time_steps=time_steps, d_model=d_model, num_heads=num_heads, mlp_dim=mlp_dim, dropout_rate=dropout_rate, L=Lt)
        self.norm = nn.LayerNorm([len(hr_channel_names), time_steps], eps=1e-6)
        
    def forward(self, x):
        # x shape: (batch_size, lr_channels, time_steps)
        sim_out = self.sim(x)
        trm_out = self.trm(sim_out)
        out = sim_out + trm_out
        out = self.norm(out)
        return out

def reconstruction_loss(y_true, y_pred, sigma1, sigma2):
    """
    Custom loss function combining frequency domain MSE and time domain MAE
    
    Args:
        y_true: (batch_size, channels, time_steps) Ground truth EEG
        y_pred: (batch_size, channels, time_steps) Super-resolved EEG
        sigma1: nn.Parameter, learnable scalar for FMSE loss
        sigma2: nn.Parameter, learnable scalar for MAE loss
    Returns:
        Scalar loss value.
    """
    # Convert to complex tensors for FFT
    fft_true = torch.fft.fft(y_true.to(torch.float32))
    fft_pred = torch.fft.fft(y_pred.to(torch.float32))
    
    # Compute frequency domain MSE
    fmse = torch.mean(torch.abs(fft_true - fft_pred) ** 2)
    
    # Compute time domain MAE
    lmae = torch.mean(torch.abs(y_true - y_pred))
    
    # Combine losses with learnable weights
    loss = (1.0 / (2.0 * sigma1**2)) * fmse + \
           (1.0 / (2.0 * sigma2**2)) * lmae + \
           torch.log(sigma1 * sigma2)
    
    return loss

# from keras.api.layers import Input, Dense, MultiHeadAttention, Permute, Dropout, LayerNormalization, Layer
# from keras.api.models import Model, Sequential
# from keras.api.optimizers import Adam
# from keras.api.losses import MeanSquaredError
# from keras.api.metrics import MeanAbsoluteError
# from keras.api.callbacks import EarlyStopping, ModelCheckpoint
# from mne.channels import make_standard_montage, get_builtin_montages
# import keras.api.ops as ops
# import tensorflow as tf
# import numpy as np

# def generate_1d_positional_encoding(time_steps, d_model):
#     """
#     Args:
#         time_steps: Total sequence length of a given EEG epoch
#         d_model: Dimension length for the sin/cos time encoding, should be an even number
#     Returns:
#         Tensor with the shape (1, time_steps, d_model)
#     """
#     assert d_model % 2 == 0, "d_model must be even for sin/cos encoding."
#     pos_encoding = np.zeros((time_steps, d_model))
#     for pos in range(time_steps):
#         for i in range(d_model // 2):
#             div_term = np.power(10000.0, (2 * i) / d_model)
#             pos_encoding[pos, 2 * i] = np.sin(pos / div_term)
#             pos_encoding[pos, 2 * i + 1] = np.cos(pos / div_term)
#     pos_encoding = np.expand_dims(pos_encoding, axis=0)
#     return tf.constant(pos_encoding, dtype=tf.float32)

# def generate_3d_positional_encoding(channel_names, d_model, builtin_montage=None, positions=[]):
#     """
#     Args:
#         channel_names: List of names of EEG electrodes
#         d_model: Dimension length for the 3D sin/cos channel encoding, should be divisible by 3
#         builtin_montage: Name of the montage built into Python-MNE, shouldn't be used if positions is provided
#         positions: 3D coordinates of the electrodes placed on the head in the order of channel names, shouldn't be used if builtin_montage is given
#     Returns:
#         Tensor with the shape (1, len(channel_names), d_model)
#     """
#     num_channels = len(channel_names)
#     builtin_montages = get_builtin_montages()
#     if num_channels == 0:
#         raise ValueError("The number of channels must be greater than 0.")
#     if builtin_montage and positions:
#         raise ValueError("You can only use either builtin_montage or positions, not both.")
#     if not builtin_montage and positions and len(positions) != num_channels:
#         raise ValueError("The number of positions must match the number of channels.")
#     if builtin_montage and not positions and builtin_montage not in builtin_montages:
#         raise ValueError(f"Montage '{builtin_montage}' is not available. Please choose from {builtin_montages}.")
#     assert d_model % 3 == 0, "d_model must be divisible by 3."
#     d_model_per_axis = d_model // 3
#     if not builtin_montage and positions and len(positions) == num_channels:
#         positions = np.array(positions)
#     if builtin_montage and not positions and builtin_montage in builtin_montages:
#         builtin_montage = make_standard_montage(builtin_montage)
#         pos_dict = builtin_montage.get_positions()['ch_pos']
#         positions = np.array([pos_dict[ch] for ch in channel_names])
#     pos_encoding = []
#     for axis in range(3):
#         pos = positions[:, axis]
#         pe = np.zeros((num_channels, d_model_per_axis))
#         for i in range(d_model_per_axis):
#             div_term = np.power(10000.0, (2 * i) / d_model_per_axis)
#             pe[:, i] = np.where(i % 2 == 0, np.sin(pos / div_term), np.cos(pos / div_term))
#         pos_encoding.append(pe)
#     pos_encoding = np.concatenate(pos_encoding, axis=-1)
#     pos_encoding = np.expand_dims(pos_encoding, axis=0)
#     return tf.constant(pos_encoding, dtype=tf.float32)

# class MaskTokenExpander(Layer):
#     """Custom layer to properly handle mask token expansion for batch dimensions"""
#     def __init__(self, mask_token, **kwargs):
#         super().__init__(**kwargs)
#         self.mask_token = mask_token
    
#     def call(self, inputs):
#         batch_size = ops.shape(inputs)[0]
#         return ops.repeat(self.mask_token, batch_size, axis=0)

# def MaskTokensInsert(lr_channel_names, d_model, hr_channel_names, mask_token):
#     """
#     Args:
#         lr_tensor: KerasTensor of shape (batch_size, lr_channels, time_steps)
#         lr_channel_names: list of low-res EEG channel names
#         hr_channel_names: list of target high-res EEG channel names
#         mask_token: Tensor of shape (1, 1, d_model), will be broadcast across batch
#     Returns:
#         KerasTensor of shape (batch_size, hr_channels, time_steps)
#     """
#     inp = Input((len(lr_channel_names), d_model))
#     lr_dict = {ch: i for i, ch in enumerate(lr_channel_names)}
#     out_list = []
#     mask_expander = MaskTokenExpander(mask_token)
#     batch_reference = inp[:, 0:1, :]
#     for ch in hr_channel_names:
#         if ch in lr_dict:
#             idx = lr_dict[ch]
#             ch_tensor = ops.expand_dims(inp[:, idx, :], axis=1)
#             out_list.append(ch_tensor)
#         else:
#             expanded_mask = mask_expander(batch_reference)
#             out_list.append(expanded_mask)
#     out = ops.concatenate(out_list, axis=1)
#     return Model(inputs=inp, outputs=out, name="MaskTokensInsert")

# def reconstruction_loss(y_true, y_pred, sigma1, sigma2):
#     """
#     Args:
#         y_true: (batch_size, channels, time_steps) Ground truth EEG
#         y_pred: (batch_size, channels, time_steps) Super-resolved EEG
#         sigma1: tf.Variable, learnable scalar for FMSE loss
#         sigma2: tf.Variable, learnable scalar for MAE loss
#     Returns:
#         Scalar loss value.
#     """
#     fft_true = tf.signal.fft(tf.cast(y_true, tf.complex64))
#     fft_pred = tf.signal.fft(tf.cast(y_pred, tf.complex64))
#     fmse = tf.reduce_mean(tf.math.abs(fft_true - fft_pred) ** 2)
#     lmae = tf.reduce_mean(tf.math.abs(y_true - y_pred))
#     loss = (1.0 / (2.0 * tf.square(sigma1))) * fmse + \
#            (1.0 / (2.0 * tf.square(sigma2))) * lmae + \
#            tf.math.log(sigma1 * sigma2)
#     return loss

# # Self Attention Block (Spatial/Temporal)
# def SAB(num_channels, time_steps, num_heads, mlp_dim, name, spatial_or_temporal="spatial", dropout_rate=0.1, L=1):
#     inp = Input(shape=(num_channels, time_steps))
#     embed_dim = time_steps if spatial_or_temporal == "spatial" else num_channels
#     out = inp
#     is_temporal = spatial_or_temporal == "temporal"
#     if is_temporal:
#         out = Permute((2, 1))(out)
#     for _ in range(L):
#         dropout1 = Dropout(dropout_rate)
#         norm1 = LayerNormalization(epsilon=1e-6)
#         norm2 = LayerNormalization(epsilon=1e-6)
#         attn = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate)
#         mlp = Sequential([
#             Dense(mlp_dim, activation='gelu'),
#             Dropout(dropout_rate),
#             Dense(embed_dim),
#             Dropout(dropout_rate)               
#         ])
#         out_norm = norm1(out)
#         attn_output = attn(out_norm, out_norm)
#         out1 = out + dropout1(attn_output)
#         out1_norm = norm2(out1)
#         mlp_output = mlp(out1_norm)
#         out2 = out1 + mlp_output
#         out += out2
#     if is_temporal:
#         out = Permute((2, 1))(out)
#     return Model(inputs=inp, outputs=out, name=name)

# # Cross Attention Block
# def CAB(num_channels, time_steps, num_heads, mlp_dim, name, dropout_rate=0.1, L=1):
#     inp = Input(shape=(num_channels, time_steps))
#     out = inp
#     for i in range(L):
#         tsab1 = SAB(name=f"CAB_TSAB1_{i+1}", num_channels=num_channels, time_steps=time_steps, num_heads=num_heads, mlp_dim=mlp_dim, spatial_or_temporal="temporal", dropout_rate=dropout_rate)
#         ssab  = SAB(name=f"CAB_SSAB_{i+1}",  num_channels=num_channels, time_steps=time_steps, num_heads=num_heads, mlp_dim=mlp_dim, spatial_or_temporal="spatial",  dropout_rate=dropout_rate)
#         tsab2 = SAB(name=f"CAB_TSAB2_{i+1}", num_channels=num_channels, time_steps=time_steps, num_heads=num_heads, mlp_dim=mlp_dim, spatial_or_temporal="temporal", dropout_rate=dropout_rate)
#         out1 = tsab1(out)
#         out1 = out + out1
#         out2 = ssab(out1)
#         out2 = out1 + out2
#         out3 = tsab2(out2)
#         out3 = out2 + out3
#         out += out3
#     return Model(inputs=inp, outputs=out, name=name)

# # Spatial Interpolation Module
# def SIM(lr_channel_names, hr_channel_names, mask_token, time_steps, d_model, num_heads, mlp_dim, name, dropout_rate=0.1, builtin_montage=None, positions=[], L=1):
#     low_res_ch_count = len(lr_channel_names)
#     high_res_ch_count = len(hr_channel_names)
#     low_res_3d_pos_encoding  = generate_3d_positional_encoding(channel_names=lr_channel_names,  d_model=d_model, builtin_montage=builtin_montage, positions=positions)
#     high_res_3d_pos_encoding = generate_3d_positional_encoding(channel_names=hr_channel_names, d_model=d_model, builtin_montage=builtin_montage, positions=positions)
#     norm1 = LayerNormalization(epsilon=1e-6)
#     norm2 = LayerNormalization(epsilon=1e-6)
#     dense1 = Dense(d_model, activation='gelu')
#     dense2 = Dense(d_model, activation='gelu')
#     dense3 = Dense(time_steps, activation='gelu')
#     cab1 = CAB(name="SIM_CAB1", num_channels=low_res_ch_count,  time_steps=d_model, num_heads=num_heads, mlp_dim=mlp_dim, dropout_rate=dropout_rate, L=L)
#     cab2 = CAB(name="SIM_CAB2", num_channels=high_res_ch_count, time_steps=d_model, num_heads=num_heads, mlp_dim=mlp_dim, dropout_rate=dropout_rate, L=L)
#     inp = Input(shape=(len(lr_channel_names), time_steps))
#     out = inp
#     out = dense1(out)
#     out = out + low_res_3d_pos_encoding
#     out = cab1(out)
#     out = norm1(out)
#     out = MaskTokensInsert(lr_channel_names, d_model, hr_channel_names, mask_token)(out) # Concatenate(axis=1)([out, mask_token])
#     out = dense2(out)
#     out = out + high_res_3d_pos_encoding
#     out = cab2(out)
#     out = norm2(out)
#     out = dense3(out)
#     return Model(inputs=inp, outputs=out, name=name)

# # Temporal Resolution Module
# def TRM(num_channels, time_steps, d_model, num_heads, mlp_dim, name, dropout_rate=0.1, L=1):
#     norm1 = LayerNormalization(epsilon=1e-6)
#     norm2 = LayerNormalization(epsilon=1e-6)
#     dense1 = Dense(d_model, activation='gelu')
#     dense2 = Dense(d_model, activation='gelu')
#     dense3 = Dense(num_channels, activation='gelu')
#     _1d_pos_encoding = generate_1d_positional_encoding(time_steps, d_model)
#     tsab1 = SAB(name="TRM_TSAB1", num_channels=d_model, time_steps=time_steps, num_heads=num_heads, mlp_dim=mlp_dim, spatial_or_temporal="temporal", dropout_rate=dropout_rate, L=L)
#     tsab2 = SAB(name="TRM_TSAB2", num_channels=d_model, time_steps=time_steps, num_heads=num_heads, mlp_dim=mlp_dim, spatial_or_temporal="temporal", dropout_rate=dropout_rate, L=L)
#     inp = Input(shape=(num_channels, time_steps))
#     out = inp
#     out = Permute((2, 1))(inp)
#     out = dense1(out)
#     out = out + _1d_pos_encoding
#     out = Permute((2, 1))(out)
#     out = tsab1(out)
#     out = norm1(out)
#     out = Permute((2, 1))(out)
#     out = dense2(out)
#     out = out + _1d_pos_encoding
#     out = Permute((2, 1))(out)
#     out = tsab2(out)
#     out = norm2(out)
#     out = Permute((2, 1))(out)
#     out = dense3(out)
#     out = Permute((2, 1))(out)
#     return Model(inputs=inp, outputs=out, name=name)

# # ESTFormer Model
# def ESTFormer(lr_channel_names, hr_channel_names, builtin_montage, time_steps, d_model, num_heads, mlp_dim, dropout_rate, Ls, Lt):
#     inp = Input(shape=(len(lr_channel_names), time_steps))
#     mask_token = tf.Variable(initial_value=tf.zeros((1, 1, d_model)), trainable=True, name="mask_token")
#     sim = SIM(name="ESTFormer_SIM", time_steps=time_steps, d_model=d_model, num_heads=num_heads, mlp_dim=mlp_dim, dropout_rate=dropout_rate, L=Ls, builtin_montage=builtin_montage, lr_channel_names=lr_channel_names, hr_channel_names=hr_channel_names, mask_token=mask_token)(inp)
#     trm = TRM(name="ESTFormer_TRM", time_steps=time_steps, d_model=d_model, num_heads=num_heads, mlp_dim=mlp_dim, dropout_rate=dropout_rate, L=Lt, num_channels=len(hr_channel_names))(sim)
#     out = sim + trm
#     out = LayerNormalization(epsilon=1e-6)(out)
#     return Model(name="ESTFormer", inputs=inp, outputs=out)