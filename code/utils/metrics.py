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