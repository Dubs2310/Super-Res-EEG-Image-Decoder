from keras.api.layers import Input, Dense, Flatten, MultiHeadAttention, Concatenate, Add, Permute, Dropout, LayerNormalization, Reshape, Layer
from keras.api.models import Model, Sequential
from keras.api.optimizers import Adam
from keras.api.losses import MeanSquaredError
from keras.api.metrics import MeanAbsoluteError
from keras.api.callbacks import EarlyStopping, ModelCheckpoint
from mne.channels import make_standard_montage, get_builtin_montages
import tensorflow as tf
import numpy as np

def gen_sab(embed_dim, num_heads, mlp_dim, name, spatial_or_temporal="spatial", dropout_rate=0.1, L=1):
    inp = Input(shape=(None, embed_dim))
    norm1 = LayerNormalization(epsilon=1e-6)
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate)
    dropout1 = Dropout(dropout_rate)
    norm2 = LayerNormalization(epsilon=1e-6)
    mlp = Sequential([
        Dense(mlp_dim, activation='gelu'),
        Dropout(dropout_rate),
        Dense(embed_dim),
        Dropout(dropout_rate)               
    ])

    out = inp
    is_temporal = spatial_or_temporal == "temporal"

    if is_temporal:
        out = Permute((2, 1))(out) # Change to (batch_size, time_steps, channels)
        
    for _ in range(L):
        out_norm = norm1(out)
        attn_output = attn(out_norm, out_norm)
        out1 = out + dropout1(attn_output)

        out1_norm = norm2(out1)
        mlp_output = mlp(out1_norm)
        out2 = out1 + mlp_output

        out += out2 # out = out + out 2
        
    if is_temporal:
        out = Permute((2, 1))(out) # Change back to (batch_size, channels, time_steps)

    return Model(inputs=inp, outputs=out, name=name)

def gen_cab(embed_dim, d_model, num_heads, mlp_dim, name, dropout_rate=0.1, L=1):
    inp = Input(shape=(None, embed_dim))

    tsab1 = gen_sab(embed_dim, num_heads, mlp_dim, "TSAB1", spatial_or_temporal="temporal", dropout_rate=dropout_rate)
    ssab = gen_sab(d_model, num_heads, mlp_dim, "SSAB", spatial_or_temporal="spatial", dropout_rate=dropout_rate)
    tsab2 = gen_sab(embed_dim, num_heads, mlp_dim, "TSAB2", spatial_or_temporal="temporal", dropout_rate=dropout_rate)

    out = inp
    for _ in range(L):
        out1 = tsab1(out)
        out1 = out + out1

        out2 = ssab(out1)
        out2 = out1 + out2

        out3 = tsab2(out2)
        out3 = out2 + out3
        out += out3

    return Model(inputs=inp, outputs=out, name=name)

def generate_3d_positional_encoding(channel_names, d_model, builtin_montage=None, positions=[]):
    num_channels = len(channel_names)
    builtin_montages = get_builtin_montages()

    if num_channels == 0:
        raise ValueError("The number of channels must be greater than 0.")
    
    if builtin_montage and positions:
        raise ValueError("You can only use either builtin_montage or positions, not both.")
    
    if not builtin_montage and positions and len(positions) != num_channels:
        raise ValueError("The number of positions must match the number of channels.")
    
    if not builtin_montage and positions and len(positions) == num_channels:
        positions = np.array(positions)

    if builtin_montage and not positions and builtin_montage not in builtin_montages:
        raise ValueError(f"Montage '{builtin_montage}' is not available. Please choose from {builtin_montages}.")
    
    if builtin_montage and not positions and builtin_montage in builtin_montages:
        builtin_montage = make_standard_montage(builtin_montage)
        pos_dict = builtin_montage.get_positions()['ch_pos']
        positions = np.array([pos_dict[ch] for ch in channel_names])  # shape: (num_channels, 3)

    assert d_model % 3 == 0, "d_model must be divisible by 3."
    d_model_per_axis = d_model // 3

    pos_encoding = []

    for axis in range(3):
        pos = positions[:, axis]
        pe = np.zeros((num_channels, d_model_per_axis))
        for i in range(d_model_per_axis):
            div_term = np.power(10000.0, (2 * i) / d_model_per_axis)
            pe[:, i] = np.where(i % 2 == 0, np.sin(pos / div_term), np.cos(pos / div_term))
    
        pos_encoding.append(pe)

    pos_encoding = np.concatenate(pos_encoding, axis=-1)    # shape: (num_channels, d_model)
    pos_encoding = np.expand_dims(pos_encoding, axis=0)     # shape: (1, num_channels, d_model)
    return tf.constant(pos_encoding, dtype=tf.float32)      # shape: (1, num_channels, d_model)

def gen_sim(embed_dim, d_model, num_heads, mlp_dim, low_res_ch_names, high_res_ch_names, name, dropout_rate=0.1, builtin_montage=None, positions=[], L=1):
    dense1 = Dense(d_model, activation='gelu')
    low_res_3d_pos_encoding = generate_3d_positional_encoding(low_res_ch_names, d_model, builtin_montage=builtin_montage, positions=positions)
    cab1 = gen_cab(embed_dim, d_model, num_heads, mlp_dim, name="CAB1", dropout_rate=dropout_rate, L=L)
    norm1 = LayerNormalization(epsilon=1e-6)
    mask_token = tf.Variable(initial_value=tf.zeros([1, d_model]), trainable=True)
    dense2 = Dense(d_model, activation='gelu')
    high_res_3d_pos_encoding = generate_3d_positional_encoding(high_res_ch_names, d_model, builtin_montage=builtin_montage, positions=positions)
    cab2 = gen_cab(embed_dim, d_model, num_heads, mlp_dim, name="CAB2", dropout_rate=dropout_rate, L=L)
    norm2 = LayerNormalization(epsilon=1e-6)

    inp = Input(shape=(None, embed_dim))
    out = inp                                       # channel embedding
    out = dense1(out)
    out = out + low_res_3d_pos_encoding
    
    out = cab1.generate(out)
    out = norm1(out)                            # feature projection
    
    out = Concatenate([out, mask_token])
    
    out = dense2(out)
    out = out + high_res_3d_pos_encoding

    out = cab2.generate(out)
    out = norm2(out)

    return Model(inputs=inp, outputs=out, name=name)

def generate_1d_positional_encoding(time_steps, d_model):
    assert d_model % 2 == 0, "d_model must be even for sin/cos encoding."
    
    pos_encoding = np.zeros((time_steps, d_model))  # Shape: (time_steps, d_model)
    
    for pos in range(time_steps):
        for i in range(d_model // 2):
            div_term = np.power(10000.0, (2 * i) / d_model)
            pos_encoding[pos, 2 * i] = np.sin(pos / div_term)
            pos_encoding[pos, 2 * i + 1] = np.cos(pos / div_term)

    pos_encoding = np.expand_dims(pos_encoding, axis=0)  # Shape: (1, time_steps, d_model)
    return tf.constant(pos_encoding, dtype=tf.float32)

def gen_trm(embed_dim, d_model, num_heads, mlp_dim, name, dropout_rate=0.1, L=1):
    dense1 = Dense(d_model, activation='gelu')
    _1d_pos_encoding = generate_1d_positional_encoding()
    tsab1 = gen_sab(embed_dim, num_heads, mlp_dim, name="TSAB21", spatial_or_temporal="temporal", dropout_rate=dropout_rate, L=L)
    norm1 = LayerNormalization(epsilon=1e-6)

    dense2 = Dense(d_model, activation='gelu')
    tsab2 = gen_sab(embed_dim, num_heads, mlp_dim, name="TSAB22", spatial_or_temporal="temporal", dropout_rate=dropout_rate, L=L)
    norm2 = LayerNormalization(epsilon=1e-6)

    inp = Input(shape=(None, embed_dim))

    out = Permute((2, 1))(inp)                      # time embedding (batch_size, time_steps, channels)
    out = dense1(out)
    out = out + _1d_pos_encoding
    out = tsab1(out)
    out = norm1(out)                            # feature projection
    out = dense2(out)
    out = out + _1d_pos_encoding
    out = tsab2(out)
    out = norm2(out)                            # feature projection
    out = Permute((2, 1))(out)                      # reshape to (batch_size, channels, time_steps)
    
    return Model(inputs=inp, outputs=out, name=name)

