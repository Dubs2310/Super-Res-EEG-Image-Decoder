import numpy as np
import tensorflow as tf
from keras.api.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Add, Concatenate, Permute
from keras.api.models import Model, Sequential
import mne

# --- Positional Encoding for Time (sinusoidal) ---
def get_temporal_positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.constant(angle_rads, dtype=tf.float32)

# --- 3D Electrode Position Lookup via MNE ---
def get_electrode_positions(channel_names, units):
    montage = mne.channels.make_standard_montage('standard_1020')
    pos_dict = montage.get_positions()['ch_pos']
    coords = np.array([pos_dict[ch] for ch in channel_names])  # shape: (num_channels, 3)
    return Dense(units)(tf.constant(coords, dtype=tf.float32))  # learnable embedding from 3D → units

# --- MLP Block ---
def mlp_block(units):
    return Sequential([
        Dense(units, activation='gelu'),
        Dense(units)
    ])

# --- TSAB Block (Temporal) ---
def tsab_block(seq_len, units, name):
    input_tensor = Input(shape=(seq_len, units))
    x_norm = LayerNormalization()(input_tensor)
    attn = MultiHeadAttention(num_heads=4, key_dim=units // 4)(x_norm, x_norm)
    x = Add()([input_tensor, attn])
    x_norm = LayerNormalization()(x)
    mlp_out = mlp_block(units)(x_norm)
    output = Add()([x, mlp_out])
    return Model(inputs=input_tensor, outputs=output, name=name)

# --- SSAB Block (Spatial) with MNE 3D embedding ---
def ssab_block(num_channels, units, name):
    input_tensor = Input(shape=(num_channels, units))
    x_norm = LayerNormalization()(input_tensor)
    attn = MultiHeadAttention(num_heads=4, key_dim=units // 4)(x_norm, x_norm)
    x = Add()([input_tensor, attn])
    x_norm = LayerNormalization()(x)
    mlp_out = mlp_block(units)(x_norm)
    output = Add()([x, mlp_out])
    return Model(inputs=input_tensor, outputs=output, name=name)

# --- CAB Block (Spatial → Temporal → Spatial) ---
def cab_block(num_channels, units):
    input_tensor = Input(shape=(num_channels, units))
    x = ssab_block(num_channels, units, name="SSAB1")(input_tensor)
    x = tsab_block(num_channels, units, name="TSAB3")(x)
    x = ssab_block(num_channels, units, name="SSAB2")(x)
    return Model(inputs=input_tensor, outputs=x, name='CAB')

# --- Spatial Interpolation Module ---
def spatial_module(input_shape, units, channel_names):
    input_tensor = Input(shape=input_shape)  # (time, channels)
    x = Permute((2, 1))(input_tensor)  # Permute to (channels, time)
    x = Dense(units)(x)
    spatial_embed = tf.expand_dims(get_electrode_positions(channel_names, units), axis=0)  # Get learnable position embedding from 3D coords # shape: (num_channels, units)
    x = Add()([x, spatial_embed])  # broadcast over batch dimension
    x = cab_block(input_shape[1], units)(x)
    x = LayerNormalization()(x)
    return Model(inputs=input_tensor, outputs=x, name='SpatialModule')

# --- Temporal Reconstruction Module ---
def temporal_module(input_shape, units):
    input_tensor = Input(shape=input_shape)  # (time, channels)
    x = Permute((2, 1))(input_tensor)  # Permute to (channels, time)
    x = Dense(units)(x)
    x = Add()([x, tf.expand_dims(get_temporal_positional_encoding(input_shape[1], units), axis=0)])
    x = tsab_block(input_shape[1], units, name="TSAB1")(x)
    x = tsab_block(input_shape[1], units, name="TSAB2")(x)
    x = LayerNormalization()(x)
    return Model(inputs=input_tensor, outputs=x, name='TemporalModule')

# --- Full ESTformer EEG Super-Resolution Model ---
def build_estformer_model(channel_names, time_samples=53, num_output_channels=64, units=128):
    '''
    Build the ESTformer model for EEG super-resolution.
    The model consists of a spatial module and a temporal module, both of which are based on the Transformer architecture.
    The spatial module uses a learnable embedding of the 3D electrode positions, while the temporal module uses a sinusoidal positional encoding.
    channel_names: list of channel names (e.g., ['Fp1', 'Fp2', ...])
    time_samples: number of time samples in the input data
    num_output_channels: number of output channels (e.g., 64 for 64-channel EEG data)
    units: number of units in the hidden layers
    '''
    input_shape=(time_samples, len(channel_names))
    input_tensor = Input(shape=input_shape[-1::-1]) # our generator gives this 14, 53, our model needs 53, 14
    reshaped_layer = Permute((2, 1))(input_tensor)
    spatial_out = spatial_module(input_shape, units, channel_names)(reshaped_layer)
    temporal_out = temporal_module(input_shape, units)(reshaped_layer)

    combined = Concatenate(axis=-1)([spatial_out, temporal_out])
    output = Dense(num_output_channels)(combined)
    
    return Model(inputs=input_tensor, outputs=output, name='ESTformer_EEG_Model')
