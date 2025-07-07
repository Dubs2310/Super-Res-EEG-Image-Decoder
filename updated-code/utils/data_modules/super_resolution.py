from utils.data_modules.base import EEGDataModule
from utils.datasets.super_resolution import EEGSuperResolutionDataset

# eeg_input_params = {
#     'eeg_montage': 'standard_1020',
#     'input_channels': ['Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4', 'FT8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2'],
#     'resampling_frequency': 250,
#     'window_before_event_ms': 50,
#     'window_after_event_ms': 600,
#     'read_eeg_dir': "S:\\PolySecLabProjects\\eeg-image-decoding\\data\\all-joined-1\\eeg\\preprocessed",
#     'save_epochs_dir': "S:\\PolySecLabProjects\\eeg-image-decoding\\data\\all-joined-1\\eeg\\epochs"
# }

# ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']

class EEGSuperResolutionDataModule(EEGDataModule):
    def __init__(self, input_channels, output_channels, sfreq, window_before_event_ms, window_after_event_ms, montage=None, eeg_dir=None, epochs_dir=None, subject=None, session=None, batch_size=32, num_workers=4, val_split=0.1):
        self.output_channels = output_channels
        self.base_datamodule_params = {
            'dataset_class': EEGSuperResolutionDataset,
            'input_channels': input_channels,
            'sfreq': sfreq,
            'window_before_event_ms': window_before_event_ms,
            'window_after_event_ms': window_after_event_ms,
            'montage': montage,
            'eeg_dir': eeg_dir,
            'epochs_dir': epochs_dir,
            'subject': subject,
            'session': session,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'val_split': val_split
        }
        super().__init__(**self.base_datamodule_params)

    def get_dataset_output_params(self, df, split='train'):
        return { 'output_channels': self.output_channels }
    
    def get_output_sample_info(self, output_sample):
        return {
            'num_channels': output_sample.shape[0],
            'num_timesteps': output_sample.shape[1],
            'channel_names': self.output_channels,
            'sfreq': self.base_datamodule_params['sfreq'],
            'epoch_window_ms': self.base_datamodule_params['window_before_event_ms'] + self.base_datamodule_params['window_after_event_ms'],
            'montage': self.base_datamodule_params['montage']
        }