import re
import os
import mne
import h5py
import numpy as np
from math import ceil
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class HDF5DataSplitGenerator(Dataset):
    def __init__(self, h5_file_path=None, dataset_type="train", dataset_split="70/25/5", random_state=97, eeg_epoch_mode="fixed_length", fixed_length_duration=8, duration_before_onset=0.05, duration_after_onset=0.6, lr_channel_names=None, hr_channel_names=None, data_dir=None, create_if_not_exists=True):
        """
        Initialize the data generator

        Args:
        
        h5_file_path: str or None
            Path to h5 file. If None, defaults to 'comprehensive_dataset.h5' in the data directory

        dataset_type: str
            Type of dataset
                Options: 'train' | 'test' | 'val'

        dataset_split: float
            Percentage of data to use for training, validation, and testing
                Format: 'train/validation/test' (e.g., '70/25/5') for 70% train, 25% validation, and 5% test or 'train/test' (e.g., '80/20') for 80% train and 20% test
                Note: The sum of the percentages must equal 100
        
        random_state: int
            Random state for train/validation/test splits
        
        eeg_epoch_mode: str
            Mode of epoch event
                Options: 'fixed_length' | 'around_evoked_event'

        fixed_length_duration: int
            This is the length of the epoch for fixed length epochs. Duration in seconds.

        duration_before_onset: int
            Duration (in seconds) before the evoked event's onset

        duration_after_onset: int
            Duration (in seconds) after the evoked event's onset
        
        lr_channel_names: list or None
            List of low-resolution channel names to use (if None, use all channels)

        hr_channel_names: list or None
            List of high-resolution channel names to use (if output type is super-resolution)
            
        data_dir: str or None
            Directory containing the raw/preprocessed data. If None, defaults to standard directory structure
            
        create_if_not_exists: bool
            Whether to create the dataset if it doesn't exist in the h5 file
        """
        
        if eeg_epoch_mode not in ['fixed_length', 'around_evoked_event']:
            raise ValueError("eeg_epoch_mode must be one of ['fixed_length', 'around_evoked_event'']")
        
        if fixed_length_duration < 1 or duration_after_onset < 0 or duration_before_onset < 0:
            raise ValueError("durations cannot be negative or 0")
        
        if dataset_type not in ['train', 'test', 'val']:
            raise ValueError("dataset_type must be one of ['train', 'test', 'val']")
        
        # use regex to check if the dataset_split is in the format 'train/test' or 'train/val/test'
        if not dataset_split or not isinstance(dataset_split, str) or not re.match(r'^\d+/\d+(/\d+)?$', dataset_split):
            raise ValueError("dataset_split must be provided in the format 'train/test' or 'train/val/test'")
        
        parts = list(map(int, dataset_split.split('/')))
        
        if sum(parts) != 100: 
            raise ValueError("Dataset split percentages must sum to 100")

        if len(parts) == 2 and dataset_type == "val":
            print("Warning: dataset_type is 'val' but only 2 splits provided. Using the first split as train and the second as test.")
        
        # Need to defer the channel validation until after we have access to the file
        # We'll store these for later validation
        self.lr_channel_names = lr_channel_names
        self.hr_channel_names = hr_channel_names

        self.eeg_epoch_mode = eeg_epoch_mode
        self.fixed_length_duration = fixed_length_duration
        self.duration_before_onset = duration_before_onset
        self.duration_after_onset = duration_after_onset
        self.dataset_type = dataset_type
        
        # Default paths if not provided
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, "..", "..", "data", "all-joined-1") if data_dir is None else data_dir
        self.h5_file_path = os.path.join(self.data_dir, "comprehensive_dataset.h5") if h5_file_path is None else h5_file_path
        
        # Check if the h5 file exists, create it if it doesn't
        if not os.path.exists(self.h5_file_path):
            os.makedirs(os.path.dirname(self.h5_file_path), exist_ok=True)
            with h5py.File(self.h5_file_path, 'w') as f:
                pass
        
        # Generate keys for the h5 file based on the selected mode
        if self.eeg_epoch_mode == 'fixed_length':
            self.epochs_key = f"all_{fixed_length_duration}s_epochs"
        else:  # around_evoked_event
            duration = self.duration_before_onset + self.duration_after_onset
            self.epochs_key = f"all_{duration}_evoked_event_epochs"
        
        self.metadata_key = f"{self.epochs_key}_metadata"

        # Check if the dataset with the specified key exists, create it if it doesn't
        # This will create both fixed_length and around_evoked_event datasets
        if create_if_not_exists:
            self._create_dataset_if_not_exists()
        
        # Load metadata once at initialization to avoid reopening the file
        with h5py.File(self.h5_file_path, 'r') as f:
            if self.epochs_key not in f:
                raise KeyError(f"Dataset '{self.epochs_key}' does not exist in the HDF5 file. "
                            f"Set create_if_not_exists=True to create it.")
            
            self.num_epochs = f[self.epochs_key].shape[0]
            self.ch_names = [ch.decode('utf-8') for ch in f['ch_names'][()]]
            self.sfreq = f['sfreq'][()]
            
            # Pre-load all metadata to memory
            self.all_metadata = f[self.metadata_key][:]
            
            # If fixed_length mode, pre-load evoked events metadata for later use
            if self.eeg_epoch_mode == 'fixed_length':
                evoked_duration = self.duration_before_onset + self.duration_after_onset
                self.evoked_epochs_key = f"all_{evoked_duration}_evoked_event_epochs"
                self.evoked_metadata_key = f"{self.evoked_epochs_key}_metadata"
                
                if self.evoked_metadata_key in f:
                    self.all_evoked_metadata = f[self.evoked_metadata_key][:]
                else:
                    self.all_evoked_metadata = None
            else:  # around_evoked_event
                self.fixed_length_epochs_key = f"all_{fixed_length_duration}s_epochs"
                self.fixed_length_metadata_key = f"{self.fixed_length_epochs_key}_metadata"
        
        # Split indices for train/test sets
        all_indices = np.arange(self.num_epochs)
        # self.train_indices, self.test_indices = train_test_split(
        #     all_indices, test_size=test_size, random_state=random_state
        # )
        try:
            if len(parts) == 2:
                train_pct, test_pct = parts
                self.train_indices, self.test_indices = train_test_split(
                    all_indices, test_size=test_pct / 100, random_state=random_state
                )

            elif len(parts) == 3:
                train_pct, val_pct, test_pct = parts
                # Split off train
                self.train_indices, temp = train_test_split(
                    all_indices, test_size=(val_pct + test_pct) / 100, random_state=random_state
                )
                # Split remaining temp into val/test
                relative_val_pct = val_pct / (val_pct + test_pct)
                self.val_indices, self.test_indices = train_test_split(
                    temp, test_size=(1 - relative_val_pct), random_state=random_state
                )
        except ValueError as e:
            raise ValueError(f"Ensure the dataset split is in the format 'train/val/test' or 'train/test'. Separately, the fixed length duration may be dividing the data into smaller chunks than expected, leaving 0 samples for the test set. Try adjusting the fixed length duration.")
        
        # Use appropriate indices based on dataset_type
        # self.indices = self.train_indices if dataset_type == "train" else self.test_indices
        if dataset_type == "train":
            self.indices = self.train_indices
        elif dataset_type == "test":
            self.indices = self.test_indices
        elif dataset_type == "val":
            self.indices = self.val_indices
        
        # Now validate channel specifications and get channel indices
        # Case 1: Both None - all channels for both lr and hr (no validation needed)
        # Case 2: lr None, hr specified - lr should use all channels, validate hr includes all channels
        # Case 3: lr specified, hr None - hr should use all channels (no validation needed)
        # Case 4: Both specified - validate lr is subset of hr
        
        if self.lr_channel_names is None and self.hr_channel_names is not None:
            # Case 2: Verify hr contains all channels
            missing_channels = [ch for ch in self.ch_names if ch not in self.hr_channel_names]
            if missing_channels:
                raise ValueError(f"When lr_channel_names is None but hr_channel_names is specified, "
                                f"hr_channel_names must include all available channels. Missing: {missing_channels}")
        
        elif self.lr_channel_names is not None and self.hr_channel_names is not None:
            # Case 4: Verify lr is subset of hr
            missing_channels = [ch for ch in self.lr_channel_names if ch not in self.hr_channel_names]
            if missing_channels:
                raise ValueError(f"lr_channel_names must be a subset of hr_channel_names. "
                                f"Missing from hr_channel_names: {missing_channels}")
        
        # Get channel indices for low and high resolution
        self.lr_indices = None
        if self.lr_channel_names is not None:
            self.lr_indices = [self.ch_names.index(ch) for ch in self.lr_channel_names if ch in self.ch_names]
        
        self.hr_indices = None
        if self.hr_channel_names is not None:
            self.hr_indices = [self.ch_names.index(ch) for ch in self.hr_channel_names if ch in self.ch_names]


    def _create_dataset_if_not_exists(self):
        """
        Check if the datasets exist in the h5 file, and create them if they don't.
        Creates both fixed-length and around-evoked-event datasets in a single pass
        through the data files to improve efficiency.
        """
        with h5py.File(self.h5_file_path, 'r+') as f:
            # First, initialize ch_names and sfreq if they don't exist
            preprocessed_data_dir = os.path.join(self.data_dir, 'eeg', 'preprocessed')
            if not os.path.exists(preprocessed_data_dir):
                raise FileNotFoundError(f"Preprocessed data directory not found: {preprocessed_data_dir}")
            
            preprocessed_files = os.listdir(preprocessed_data_dir)
            if not preprocessed_files:
                raise FileNotFoundError('The preprocessed data directory provided has no preprocessed files.')
            
            # Initialize ch_names and sfreq if they don't exist
            if 'ch_names' not in f or 'sfreq' not in f:
                first_raw = mne.io.read_raw_fif(os.path.join(preprocessed_data_dir, preprocessed_files[0]), preload=True)
                first_raw.drop_channels(['Status'])
                sfreq = first_raw.info['sfreq']
                ch_names = first_raw.info['ch_names']
                
                # Create datasets for sfreq and ch_names if they don't exist
                if 'sfreq' not in f:
                    f.create_dataset('sfreq', data=sfreq)
                
                if 'ch_names' not in f:
                    dt = h5py.special_dtype(vlen=str)
                    f.create_dataset('ch_names', data=np.array(ch_names, dtype=dt))
            else:
                # Use existing values
                sfreq = f['sfreq'][()]
                ch_names = [ch.decode('utf-8') for ch in f['ch_names'][()]]
            
            # Define keys for both types of datasets
            fixed_length_key = f"all_{self.fixed_length_duration}s_epochs"
            fixed_length_metadata_key = f"{fixed_length_key}_metadata"
            
            evoked_duration = self.duration_before_onset + self.duration_after_onset
            evoked_event_key = f"all_{evoked_duration}_evoked_event_epochs"
            evoked_event_metadata_key = f"{evoked_event_key}_metadata"
            
            # Check if datasets need to be created and populated
            need_fixed_length = fixed_length_key not in f
            need_evoked_event = evoked_event_key not in f
            
            # If both datasets already exist, we're done
            if not need_fixed_length and not need_evoked_event:
                return
                
            # Create fixed length dataset if it doesn't exist
            if need_fixed_length:
                f.create_dataset(
                    fixed_length_key,
                    shape=(0, len(ch_names), int(sfreq * self.fixed_length_duration)),
                    maxshape=(None, len(ch_names), int(sfreq * self.fixed_length_duration)),
                    dtype=np.float32
                )
                
                f.create_dataset(
                    fixed_length_metadata_key, 
                    shape=(0, 3),  # subject, session, sample_number
                    maxshape=(None, 3),
                    dtype=np.int32
                )
                
            # Create evoked event dataset if it doesn't exist
            if need_evoked_event:
                timesteps = int(ceil(sfreq * evoked_duration))
                
                f.create_dataset(
                    evoked_event_key,
                    shape=(0, len(ch_names), timesteps),
                    maxshape=(None, len(ch_names), timesteps),
                    dtype=np.float32
                )
                
                f.create_dataset(
                    evoked_event_metadata_key, 
                    shape=(0, 4),  # subject, session, sample_number, coco_id
                    maxshape=(None, 4),
                    dtype=np.int32
                )
            
            # Now populate both datasets in a single pass through the files
            for file in preprocessed_files:
                subject = int(file[5:6])
                session = int(file[14:15])
                
                # Load the raw data once for both epoch types
                raw = mne.io.read_raw_fif(os.path.join(preprocessed_data_dir, file), preload=True)
                
                # Process fixed length epochs if needed
                if need_fixed_length:
                    fixed_epochs = mne.make_fixed_length_epochs(raw, duration=self.fixed_length_duration, preload=True)
                    fixed_epochs.drop_channels(['Status'])
                    
                    fixed_data = fixed_epochs.get_data()  # (batch size, channels, timesteps)
                    sample_numbers = fixed_epochs.events[:, 0]
                    
                    fixed_metadata = np.zeros((fixed_data.shape[0], 3), dtype=np.int32)
                    fixed_metadata[:, 0] = subject
                    fixed_metadata[:, 1] = session
                    fixed_metadata[:, 2] = sample_numbers
                    
                    # Add data to the fixed length dataset
                    current_size = f[fixed_length_key].shape[0]
                    new_size = current_size + fixed_data.shape[0]
                    f[fixed_length_key].resize(new_size, axis=0)
                    f[fixed_length_metadata_key].resize(new_size, axis=0)
                    
                    f[fixed_length_key][current_size:new_size] = fixed_data
                    f[fixed_length_metadata_key][current_size:new_size] = fixed_metadata
                
                # Process evoked event epochs if needed
                if need_evoked_event:
                    evoked_events = mne.find_events(raw)
                    if len(evoked_events) > 0:  # Only proceed if events were found
                        evoked_epochs = mne.Epochs(
                            raw, evoked_events, 
                            tmin=-self.duration_before_onset, 
                            tmax=self.duration_after_onset+0.01, 
                            preload=True
                        )
                        evoked_epochs.drop_channels(['Status'])
                        
                        timesteps = int(ceil(sfreq * evoked_duration))
                        evoked_data = evoked_epochs.get_data()[:, :, :timesteps]  # (batch size, channels, timesteps)
                        sample_numbers = evoked_events[:, 0]
                        evoked_event_ids = evoked_events[:, -1]
                        
                        n_epochs = evoked_data.shape[0]
                        evoked_metadata = np.zeros((n_epochs, 4), dtype=np.int32)
                        evoked_metadata[:, 0] = subject
                        evoked_metadata[:, 1] = session
                        evoked_metadata[:, 2] = sample_numbers[:n_epochs]
                        evoked_metadata[:, 3] = evoked_event_ids[:n_epochs]
                        
                        # Add data to the evoked event dataset
                        current_size = f[evoked_event_key].shape[0]
                        new_size = current_size + evoked_data.shape[0]
                        f[evoked_event_key].resize(new_size, axis=0)
                        f[evoked_event_metadata_key].resize(new_size, axis=0)
                        
                        f[evoked_event_key][current_size:new_size] = evoked_data
                        f[evoked_event_metadata_key][current_size:new_size] = evoked_metadata


    def __len__(self):
        """Return the number of epochs in the dataset"""
        return len(self.indices)


    def __getitem__(self, index):
        """Get a single epoch at the specified index (not a batch)"""
        if index >= len(self.indices):
            raise IndexError(f"Index {index} out of bounds for dataset of size {len(self.indices)}")
        
        actual_index = self.indices[index]
        
        # Use a new file handle each time to avoid thread-safety issues
        with h5py.File(self.h5_file_path, 'r') as f:
            # Get the epoch data - this is the only data we need to read from the file
            epoch_data = f[self.epochs_key][actual_index]
        
        # Use the metadata we preloaded during initialization
        metadata = self.all_metadata[actual_index]
        
        # Filter channels if needed
        lr_data = epoch_data[self.lr_indices] if self.lr_indices is not None else epoch_data
        hr_data = epoch_data[self.hr_indices] if self.hr_indices is not None else epoch_data
        
        # Extract metadata
        subject = metadata[0]
        session = metadata[1]
        sample_number = metadata[2]
        
        # # Prepare evoked_event_ids based on epoch mode
        # if self.eeg_epoch_mode == 'around_evoked_event':
        #     evoked_event_ids = [metadata[3]]
        # elif self.eeg_epoch_mode == 'fixed_length':
        #     # For fixed length events, find all evoked events within the epoch
        #     evoked_event_ids = []
        #     if self.all_evoked_metadata is not None:
        #         # Calculate the end sample number
        #         end_sample = sample_number + int(self.fixed_length_duration * self.sfreq)
                
        #         # Filter events that fall within our epoch's time range
        #         matching_events = self.all_evoked_metadata[
        #             (self.all_evoked_metadata[:, 0] == subject) &  # Same subject
        #             (self.all_evoked_metadata[:, 1] == session) &  # Same session
        #             (self.all_evoked_metadata[:, 2] >= sample_number) &  # Event starts after or at epoch start
        #             (self.all_evoked_metadata[:, 2] < end_sample)  # Event starts before epoch ends
        #         ]
                
        #         # Extract just the event IDs
        #         if len(matching_events) > 0:
        #             evoked_event_ids = matching_events[:, 3].tolist()

        # Return dictionary with all required info
        out = {
            "index_before_random_split": actual_index,
            "sfreq": self.sfreq,  # Use cached sampling frequency
            "mode": self.eeg_epoch_mode,
            "total_duration": self.fixed_length_duration if self.eeg_epoch_mode == 'fixed_length' else self.duration_before_onset + self.duration_after_onset, 
            "subject": subject,
            "session": session,
            "sample_number": sample_number,
            "lo_res": lr_data,
            "hi_res": hr_data,
        }

        if self.eeg_epoch_mode == 'around_evoked_event':
            out["coco_id"] = metadata[3]

        return out
    
    def get_evoked_event_metadata_for_item(self, index):
        item_metadata = self.all_metadata[index]
        subject = item_metadata[0]
        session = item_metadata[1]
        sample_number = item_metadata[2]

        if self.eeg_epoch_mode == 'around_evoked_event':
            evoked_events_metadata = [(sample_number, item_metadata[3])]
            
        elif self.eeg_epoch_mode == 'fixed_length':
            # For fixed length events, find all evoked events within the epoch
            evoked_events_metadata = []
            if self.all_evoked_metadata is not None:
                # Calculate the end sample number
                end_sample = sample_number + int(self.fixed_length_duration * self.sfreq)
                
                # Filter events that fall within our epoch's time range
                matching_evoked_metadata = self.all_evoked_metadata[
                    (self.all_evoked_metadata[:, 0] == subject) &  # Same subject
                    (self.all_evoked_metadata[:, 1] == session) &  # Same session
                    (self.all_evoked_metadata[:, 2] >= sample_number) &  # Event starts after or at epoch start
                    (self.all_evoked_metadata[:, 2] < end_sample)  # Event starts before epoch ends
                ]
                
                # Extract just the event IDs
                if len(matching_evoked_metadata) > 0:
                    evoked_events_metadata = matching_evoked_metadata[:[2, 3]].tolist()
                    
        return evoked_events_metadata
    
    # def extract_evoked_epochs_from_fixed_length(self, index):
    #     metadata = self.all_metadata[index]
    #     subject = metadata[0]
    #     session = metadata[1]
    #     sample_number = metadata[2]

    #     # For fixed length events, find all evoked events within the epoch
    #     evoked_event_ids = []
    #     if self.all_evoked_metadata is not None:
    #         # Calculate the end sample number
    #         end_sample = sample_number + int(self.fixed_length_duration * self.sfreq)
            
    #         # Filter events that fall within our epoch's time range
    #         matching_events = self.all_evoked_metadata[
    #             (self.all_evoked_metadata[:, 0] == subject) &  # Same subject
    #             (self.all_evoked_metadata[:, 1] == session) &  # Same session
    #             (self.all_evoked_metadata[:, 2] >= sample_number) &  # Event starts after or at epoch start
    #             (self.all_evoked_metadata[:, 2] < end_sample)  # Event starts before epoch ends
    #         ]
            
    #         # Extract just the event IDs
    #         if len(matching_events) > 0:
    #             evoked_event_ids = matching_events[:, 3].tolist()
        
    #     return evoked_event_ids