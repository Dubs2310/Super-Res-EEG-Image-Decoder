from keras.api.layers import Input, Dense, Flatten, MultiHeadAttention, Concatenate, Add, Permute, Dropout, LayerNormalization, Reshape, Layer
from keras.api.models import Model, Sequential
from keras.api.optimizers import Adam
from keras.api.losses import MeanSquaredError
from keras.api.metrics import MeanAbsoluteError
from keras.api.callbacks import EarlyStopping, ModelCheckpoint
from mne.channels import make_standard_montage, get_builtin_montages
import tensorflow as tf
import numpy as np

def generate_1d_positional_encoding(time_steps, d_model):
    assert d_model % 2 == 0, "d_model must be even for sin/cos encoding."
    pos_encoding = np.zeros((time_steps, d_model))
    for pos in range(time_steps):
        for i in range(d_model // 2):
            div_term = np.power(10000.0, (2 * i) / d_model)
            pos_encoding[pos, 2 * i] = np.sin(pos / div_term)
            pos_encoding[pos, 2 * i + 1] = np.cos(pos / div_term)
    pos_encoding = np.expand_dims(pos_encoding, axis=0)
    return tf.constant(pos_encoding, dtype=tf.float32)

def generate_3d_positional_encoding(channel_names, d_model, builtin_montage=None, positions=[]):
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
    return tf.constant(pos_encoding, dtype=tf.float32)

def generate_mask_token(number_of_missing_channels, d_model):
    return tf.expand_dims(tf.Variable(initial_value=tf.zeros((number_of_missing_channels, d_model)), trainable=True), axis=0)

def estformer_reconstruction_loss(y_true, y_pred, sigma1, sigma2):
    """
    Args:
        y_true: (batch_size, channels, time_steps) Ground truth EEG
        y_pred: (batch_size, channels, time_steps) Super-resolved EEG
        sigma1: tf.Variable, learnable scalar for FMSE loss
        sigma2: tf.Variable, learnable scalar for MAE loss
    Returns:
        Scalar loss value.
    """
    fft_true = tf.signal.fft(tf.cast(y_true, tf.complex64))
    fft_pred = tf.signal.fft(tf.cast(y_pred, tf.complex64))
    fmse = tf.reduce_mean(tf.math.abs(fft_true - fft_pred) ** 2)
    lmae = tf.reduce_mean(tf.math.abs(y_true - y_pred))
    loss = (1.0 / (2.0 * tf.square(sigma1))) * fmse + \
           (1.0 / (2.0 * tf.square(sigma2))) * lmae + \
           tf.math.log(sigma1 * sigma2)
    return loss

# Self Attention Block (Spatial/Temporal)
def SAB(num_channels, time_steps, num_heads, mlp_dim, name, spatial_or_temporal="spatial", dropout_rate=0.1, L=1):
    inp = Input(shape=(num_channels, time_steps))
    embed_dim = time_steps if spatial_or_temporal == "spatial" else num_channels
    out = inp
    is_temporal = spatial_or_temporal == "temporal"
    if is_temporal:
        out = Permute((2, 1))(out)
    for _ in range(L):
        dropout1 = Dropout(dropout_rate)
        norm1 = LayerNormalization(epsilon=1e-6)
        norm2 = LayerNormalization(epsilon=1e-6)
        attn = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate)
        mlp = Sequential([
            Dense(mlp_dim, activation='gelu'),
            Dropout(dropout_rate),
            Dense(embed_dim),
            Dropout(dropout_rate)               
        ])
        out_norm = norm1(out)
        attn_output = attn(out_norm, out_norm)
        out1 = out + dropout1(attn_output)
        out1_norm = norm2(out1)
        mlp_output = mlp(out1_norm)
        out2 = out1 + mlp_output
        out += out2
    if is_temporal:
        out = Permute((2, 1))(out)
    return Model(inputs=inp, outputs=out, name=name)

# Cross Attention Block
def CAB(num_channels, time_steps, num_heads, mlp_dim, name, dropout_rate=0.1, L=1):
    inp = Input(shape=(num_channels, time_steps))
    out = inp
    for i in range(L):
        tsab1 = SAB(name=f"CAB_TSAB1_{i+1}", num_channels=num_channels, time_steps=time_steps, num_heads=num_heads, mlp_dim=mlp_dim, spatial_or_temporal="temporal", dropout_rate=dropout_rate)
        ssab  = SAB(name=f"CAB_SSAB_{i+1}",  num_channels=num_channels, time_steps=time_steps, num_heads=num_heads, mlp_dim=mlp_dim, spatial_or_temporal="spatial",  dropout_rate=dropout_rate)
        tsab2 = SAB(name=f"CAB_TSAB2_{i+1}", num_channels=num_channels, time_steps=time_steps, num_heads=num_heads, mlp_dim=mlp_dim, spatial_or_temporal="temporal", dropout_rate=dropout_rate)
        out1 = tsab1(out)
        out1 = out + out1
        out2 = ssab(out1)
        out2 = out1 + out2
        out3 = tsab2(out2)
        out3 = out2 + out3
        out += out3
    return Model(inputs=inp, outputs=out, name=name)

# Spatial Interpolation Module
def SIM(low_res_ch_names, high_res_ch_names, time_steps, d_model, num_heads, mlp_dim, name, dropout_rate=0.1, builtin_montage=None, positions=[], L=1):
    low_res_ch_count = len(low_res_ch_names)
    high_res_ch_count = len(high_res_ch_names)
    missing_ch_count = high_res_ch_count - low_res_ch_count
    mask_token = generate_mask_token(missing_ch_count, d_model)
    low_res_3d_pos_encoding  = generate_3d_positional_encoding(channel_names=low_res_ch_names,  d_model=d_model, builtin_montage=builtin_montage, positions=positions)
    high_res_3d_pos_encoding = generate_3d_positional_encoding(channel_names=high_res_ch_names, d_model=d_model, builtin_montage=builtin_montage, positions=positions)
    norm1 = LayerNormalization(epsilon=1e-6)
    norm2 = LayerNormalization(epsilon=1e-6)
    dense1 = Dense(d_model, activation='gelu')
    dense2 = Dense(d_model, activation='gelu')
    dense3 = Dense(time_steps, activation='gelu')
    cab1 = CAB(name="SIM_CAB1", num_channels=low_res_ch_count,  time_steps=d_model, num_heads=num_heads, mlp_dim=mlp_dim, dropout_rate=dropout_rate, L=L)
    cab2 = CAB(name="SIM_CAB2", num_channels=high_res_ch_count, time_steps=d_model, num_heads=num_heads, mlp_dim=mlp_dim, dropout_rate=dropout_rate, L=L)
    inp = Input(shape=(len(low_res_ch_names), time_steps))
    out = inp
    out = dense1(out)
    out = out + low_res_3d_pos_encoding
    out = cab1(out)
    out = norm1(out)
    out = Concatenate(axis=1)([out, mask_token])
    out = dense2(out)
    out = out + high_res_3d_pos_encoding
    out = cab2(out)
    out = norm2(out)
    out = dense3(out)
    return Model(inputs=inp, outputs=out, name=name)

# Temporal Resolution Module
def TRM(num_channels, time_steps, d_model, num_heads, mlp_dim, name, dropout_rate=0.1, L=1):
    norm1 = LayerNormalization(epsilon=1e-6)
    norm2 = LayerNormalization(epsilon=1e-6)
    dense1 = Dense(d_model, activation='gelu')
    dense2 = Dense(d_model, activation='gelu')
    dense3 = Dense(num_channels, activation='gelu')
    _1d_pos_encoding = generate_1d_positional_encoding(time_steps, d_model)
    tsab1 = SAB(name="TRM_TSAB1", num_channels=d_model, time_steps=time_steps, num_heads=num_heads, mlp_dim=mlp_dim, spatial_or_temporal="temporal", dropout_rate=dropout_rate, L=L)
    tsab2 = SAB(name="TRM_TSAB2", num_channels=d_model, time_steps=time_steps, num_heads=num_heads, mlp_dim=mlp_dim, spatial_or_temporal="temporal", dropout_rate=dropout_rate, L=L)
    inp = Input(shape=(num_channels, time_steps))
    out = inp
    out = Permute((2, 1))(inp)
    out = dense1(out)
    out = out + _1d_pos_encoding
    out = Permute((2, 1))(out)
    out = tsab1(out)
    out = norm1(out)
    out = Permute((2, 1))(out)
    out = dense2(out)
    out = out + _1d_pos_encoding
    out = Permute((2, 1))(out)
    out = tsab2(out)
    out = norm2(out)
    out = Permute((2, 1))(out)
    out = dense3(out)
    out = Permute((2, 1))(out)
    return Model(inputs=inp, outputs=out, name=name)

# ESTFormer Model
def ESTFormer(low_res_ch_names, high_res_ch_names, builtin_montage, time_steps, d_model, num_heads, mlp_dim, dropout_rate, Ls, Lt):
    inp = Input(shape=(len(low_res_ch_names), time_steps))
    sim = SIM(name="ESTFormer_SIM", time_steps=time_steps, d_model=d_model, num_heads=num_heads, mlp_dim=mlp_dim, dropout_rate=dropout_rate, L=Ls, builtin_montage=builtin_montage, low_res_ch_names=low_res_ch_names, high_res_ch_names=high_res_ch_names)(inp)
    trm = TRM(name="ESTFormer_TRM", time_steps=time_steps, d_model=d_model, num_heads=num_heads, mlp_dim=mlp_dim, dropout_rate=dropout_rate, L=Lt, num_channels=len(high_res_ch_names))(sim)
    out = sim + trm
    return Model(name="ESTFormer", inputs=inp, outputs=out)