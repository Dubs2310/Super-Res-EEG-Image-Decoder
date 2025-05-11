import torch

def nmse(y_true, y_pred):
    mse = torch.mean((y_true - y_pred) ** 2)
    norm = torch.mean(y_true ** 2)
    return mse / norm

def mae(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

def pcc(y_true, y_pred):
    y_true = y_true - torch.mean(y_true, dim=-1, keepdim=True)
    y_pred = y_pred - torch.mean(y_pred, dim=-1, keepdim=True)
    numerator = torch.sum(y_true * y_pred, dim=-1)
    denominator = torch.sqrt(torch.sum(y_true ** 2, dim=-1) * torch.sum(y_pred ** 2, dim=-1))
    return torch.mean(numerator / (denominator + 1e-8))  # Mean over channels

def snr(y_true, y_pred):
    signal_power = torch.mean(y_true ** 2)
    noise_power = torch.mean((y_true - y_pred) ** 2)
    return 10 * torch.log10(signal_power / (noise_power + 1e-8))

def loss(y_true, y_pred, sigma1, sigma2):
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
    fft_true = torch.fft.rfft(y_true.to(torch.float32))
    fft_pred = torch.fft.rfft(y_pred.to(torch.float32))
    fmse = torch.mean(torch.abs(fft_true - fft_pred) ** 2) # Compute frequency domain MSE
    lmae = torch.mean(torch.abs(y_true - y_pred)) # Compute time domain MAE
    
    # Combine losses with learnable weights
    loss = (1.0 / (2.0 * sigma1**2)) * fmse + (1.0 / (2.0 * sigma2**2)) * lmae + torch.log(sigma1 * sigma2)
    return loss