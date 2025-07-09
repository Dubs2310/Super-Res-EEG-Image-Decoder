import os
import mne
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utils.epoch_helpers import epoch_around_events
mne.set_log_level('WARNING')

def load_config():
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    else:
        return {}

CONFIG = load_config()
DEFAULT_EEG_MONTAGE = CONFIG.get('eeg', {}).get('montage', 'standard_1020')
DEFAULT_EEG_PATH = CONFIG.get('eeg', {}).get('preprocessed_dir', "/workspace/eeg-image-decoding/data/all-joined-1/eeg/preprocessed")
DEFAULT_EPOCH_PATH = CONFIG.get('eeg', {}).get('epochs_dir', "/workspace/eeg-image-decoding/data/all-joined-1/eeg/epochs")

print(DEFAULT_EPOCH_PATH)

class EEGDataset(Dataset):
    def __init__(self, img_pair_df: pd.DataFrame, input_channels, sfreq, window_before_event_ms, window_after_event_ms, montage=None, eeg_dir=None, epochs_dir=None):
        self.input_channels = input_channels
        self.montage = DEFAULT_EEG_MONTAGE if not montage else montage
        self.sfreq = sfreq 
        self.window_before_event_ms = window_before_event_ms
        self.window_after_event_ms = window_after_event_ms
        self.eeg_dir = DEFAULT_EEG_PATH if not eeg_dir else eeg_dir
        self.epochs_dir = os.path.join(DEFAULT_EPOCH_PATH if not epochs_dir else epochs_dir, f'{window_before_event_ms + window_after_event_ms}ms-{sfreq}Hz')

        subject_session_df = img_pair_df[['subject', 'session']].drop_duplicates()
        self.eeg_file_paths = [os.path.join(self.eeg_dir, f"subj0{row['subject']}_session{row['session']}_eeg.fif") for _, row in subject_session_df.iterrows()]
        
        os.makedirs(self.epochs_dir, exist_ok=True)

        # Get the actual EEG channel names from the first file to determine what channels will be in epochs
        first_eeg_file = self.eeg_file_paths[0]
        raw_temp = mne.io.read_raw_fif(first_eeg_file, preload=False)
        picks = mne.pick_types(raw_temp.info, eeg=True, stim=False)
        self.actual_eeg_channel_names = [raw_temp.ch_names[i] for i in picks]
        
        # Calculate channel indices based on actual EEG channels that will be in epoched data
        self.channel_indices = [i for i, ch in enumerate(self.actual_eeg_channel_names) if ch in self.input_channels]
        
        # print(f"Total EEG channels in epoched data: {len(self.actual_eeg_channel_names)}")
        # print(f"Using {len(self.channel_indices)} channels for input")
        # print(f"Selected channels: {[self.actual_eeg_channel_names[i] for i in self.channel_indices]}")
        
        # Track which epochs to drop from the dataframe
        dropped_records = []
        
        for _, row in subject_session_df.iterrows():
            subject = row['subject']
            session = row['session']
            epoch_file = self._get_epoch_filepath(subject, session)
            dropped_file = self._get_dropped_filepath(subject, session)

            if os.path.exists(epoch_file) and os.path.exists(dropped_file):
                # Load existing dropped indices
                dropped_epoch_indices = np.load(dropped_file)
                for dropped_idx in dropped_epoch_indices:
                    dropped_records.append((subject, session, dropped_idx))
                continue

            eeg_file = os.path.join(self.eeg_dir, f"subj0{subject}_session{session}_eeg.fif")
            raw = mne.io.read_raw_fif(eeg_file, preload=True)
            
            # Use your epoch_around_events function
            tmin = window_before_event_ms / 1000   # Make this positive
            tmax = window_after_event_ms / 1000    # Keep this positive
            epochs, dropped_epoch_indices = epoch_around_events(raw, tmin, tmax, resample=sfreq)
            epochs_data = epochs.get_data()
            
            # Save both epochs and dropped indices
            np.save(epoch_file, epochs_data)
            np.save(dropped_file, dropped_epoch_indices)
            
            # Track which records to drop from main dataframe
            for dropped_idx in dropped_epoch_indices:
                dropped_records.append((subject, session, dropped_idx))
            
            # print(f"Subject {subject}, Session {session}: {len(epochs_data)} valid epochs, {len(dropped_epoch_indices)} dropped")
            
            del raw, epochs, epochs_data

        # Filter the main dataframe to remove dropped epochs
        print(f"Original dataframe size: {len(img_pair_df)}")
        
        if dropped_records:
            # Create a mask for records to keep
            keep_mask = np.ones(len(img_pair_df), dtype=bool)
            
            for subject, session, epoch_idx in dropped_records:
                mask = (img_pair_df['subject'] == subject) & (img_pair_df['session'] == session) & (img_pair_df['epoch_idx'] == epoch_idx)
                keep_mask &= ~mask
            
            self.df = img_pair_df[keep_mask].reset_index(drop=True)
            # print(f"Dropped {len(dropped_records)} records due to bad epochs")
        else:
            self.df = img_pair_df.copy()
        
        # print(f"Final dataframe size: {len(self.df)}")

    def _load_eeg_epoch(self, row):
        subject = row['subject']
        session = row['session']
        epoch_idx = row['epoch_idx']
        epoch_file = self._get_epoch_filepath(subject, session)
        epochs_data = np.load(epoch_file)
        epoch = epochs_data[epoch_idx]
        
        # Use the pre-calculated channel indices based on actual EEG channels
        epoch = epoch[self.channel_indices, :]
        return torch.tensor(epoch, dtype=torch.float32)

    def _get_epoch_filepath(self, subject, session):
        return os.path.join(self.epochs_dir, f'subj0{subject}_session{session}_epochs.npy')
    
    def _get_dropped_filepath(self, subject, session):
        return os.path.join(self.epochs_dir, f'subj0{subject}_session{session}_dropped_epochs.csv')
    
    def _get_output(self, epoch, row):
        raise NotImplementedError

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        epoch = self._load_eeg_epoch(row)
        output = self._get_output(epoch, row)
        return epoch, output