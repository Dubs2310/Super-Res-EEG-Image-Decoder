import torch
import pandas as pd
from utils.datasets.base import EEGDataset

class EEGModelPredictionDataset(EEGDataset):
    def __init__(self, model, img_pair_df: pd.DataFrame, input_channels, sfreq, window_before_event_ms, window_after_event_ms, montage=None, eeg_dir=None, epochs_dir=None):
        self.model = model
        self.model.eval()
        super().__init__(self, img_pair_df, input_channels, sfreq, window_before_event_ms, window_after_event_ms, montage, eeg_dir, epochs_dir)
        # Save outputs of epoch model prediction under epochs same folder
    
    def _get_output(self, epoch, row):
        channel_indices = [i for i, ch in enumerate(self.channel_names) if ch in self.eeg_input_channels]
        input_epoch = epoch[channel_indices, :]
        input_tensor = torch.tensor(input_epoch, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)
        return output.squeeze(0)