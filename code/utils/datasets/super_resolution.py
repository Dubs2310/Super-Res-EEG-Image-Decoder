import torch
import numpy as np
import pandas as pd
from utils.datasets.base import EEGDataset

class EEGSuperResolutionDataset(EEGDataset):
    def __init__(self, img_pair_df: pd.DataFrame, input_channels, output_channels, sfreq, window_before_event_ms, window_after_event_ms, montage=None, eeg_dir=None, epochs_dir=None):
        self.output_channels = output_channels
        super().__init__(img_pair_df, input_channels, sfreq, window_before_event_ms, window_after_event_ms, montage, eeg_dir, epochs_dir)
        self.output_channel_indices = [i for i, ch in enumerate(self.actual_eeg_channel_names) if ch in self.output_channels]
        
    def _get_output(self, epoch, row):
        subject = row['subject']
        session = row['session']
        epoch_idx = row['epoch_idx']
        epoch_file = self._get_epoch_filepath(subject, session)
        epochs_data = np.load(epoch_file)
        full_epoch = epochs_data[epoch_idx]
        
        # Select output channels from the full epoch
        output_epoch = full_epoch[self.output_channel_indices, :]
        return torch.tensor(output_epoch, dtype=torch.float32)