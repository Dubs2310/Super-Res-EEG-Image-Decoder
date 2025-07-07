import re
import os
import mne
import h5py
import math
import numpy as np
from math import ceil
from torch.utils.data import Dataset
from utils.coco_data_handler import COCODataHandler
from sklearn.model_selection import train_test_split
from utils.epoch_manipulation_helpers import epoch_around_events, epoch_fixed_lengths

class EpochDataReader(Dataset):
    def __init__(self, subject_session_id='cross', read_from="ground-truth", channel_names=None, resample_freq=512, epoch_type="around_evoked", before=0.05, after=0.6, fixed_length_duration=8, split_type=None, split="70/25/5", random_state=97):
        self.subject_session_id = subject_session_id
        self.read_from = read_from
        self.channel_names = channel_names
        self.resample_freq = resample_freq
        self.epoch_type = epoch_type
        self.before = before
        self.after = after
        self.fixed_length_duration = fixed_length_duration
        self.split_type = split_type
        self.split = split
        self.random_state = random_state
        
        if read_from not in ['ground-truth', 'super-resolution']:
            raise ValueError("read_from must be either 'ground-truth' or 'super-resolution'")
        
        
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "all-joined-1", "eeg")
        data_dir = os.path.join(base_dir, 'preprocessed')
        parts = list(map(int, split.split('/')))

        # Initialize one_hot_encodings to None
        self.one_hot_encodings = None
        self.h5_path = os.path.join(base_dir, 'cross-subjects.h5')

        if subject_session_id == 'cross':
            # Try to handle corrupted files
            try:
                f = h5py.File(self.h5_path, 'a')
            except Exception as e:
                print(f"Error with HDF5 file: {str(e)}. Creating new file.")
                if os.path.exists(self.h5_path):
                    os.remove(self.h5_path)
                f = h5py.File(self.h5_path, 'a')
            
            group_name = [
                subject_session_id,
                read_from,
                '-'.join(sorted(channel_names)) if channel_names else 'all',
                str(resample_freq) if resample_freq else 'original',
                epoch_type,
                str(before+after) if epoch_type == "around_evoked" else str(fixed_length_duration),
                split.replace('/','_'),
                str(random_state)
            ]

            self.group_name = '/'.join(group_name)
            
            # Check if group exists using try/except instead of "in"
            try:
                group = f[self.group_name]
                
                # Store shapes and dataset info for later use
                self.epochs_shape = group['epochs'].shape
                self.has_encodings = 'one-hot-encodings' in group
                
                # Load small dataset entirely into memory for speed
                if self.epochs_shape[0] < 1000:  # Only for reasonably sized datasets
                    print(f"Loading small dataset ({self.epochs_shape[0]} items) into memory")
                    self.epochs_data = np.array(group['epochs'])
                    if self.has_encodings:
                        self.one_hot_encodings = np.array(group['one-hot-encodings'])
                else:
                    # For large datasets, don't load into memory
                    self.epochs_data = None
                    
            except Exception:
                print(f"Creating new group: {self.group_name}")
                preprocessed_files = os.listdir(os.path.join(data_dir, read_from))
                raw = mne.io.read_raw_fif(os.path.join(data_dir, read_from, preprocessed_files[0]))
                
                if channel_names is None:
                    channel_names = [ch for ch in raw.info['ch_names'] if ch != 'Status']
                    
                sfreq = raw.info['sfreq'] if not resample_freq else resample_freq
                timesteps = int(ceil(sfreq * (before + after + 0.01)) if epoch_type == "around_evoked" else (sfreq * fixed_length_duration))

                group = f.create_group(self.group_name)
                epochs_dset = group.create_dataset(
                    'epochs',
                    shape=(0, len(channel_names), timesteps),
                    maxshape=(None, len(channel_names), timesteps),
                    dtype=np.float32
                )
                
                if epoch_type == 'around_evoked':
                    ohe_dset = group.create_dataset(
                        'one-hot-encodings',
                        shape=(0, len(COCODataHandler.get_instance().category_index.keys())),
                        maxshape=(None, len(COCODataHandler.get_instance().category_index.keys())),
                        dtype=np.float32
                    )
            
                # Process files and add data to the dataset
                for eeg in preprocessed_files:
                    try:
                        raw = mne.io.read_raw_fif(os.path.join(data_dir, read_from, eeg))
                        epo, ohe = epoch_around_events(raw, before, after, channel_names, resample_freq) if epoch_type == 'around_evoked' else epoch_fixed_lengths(raw, fixed_length_duration, channel_names, resample_freq)
                        epo = epo.get_data()
                        
                        current_size = epochs_dset.shape[0]
                        new_size = current_size + epo.shape[0]
                        epochs_dset.resize(new_size, axis=0)
                        epochs_dset[current_size:new_size] = epo
                        
                        if ohe is not None and epoch_type == 'around_evoked':
                            ohe_dset.resize(new_size, axis=0)
                            ohe_dset[current_size:new_size] = ohe
                    except Exception as e:
                        print(f"Error processing {eeg}: {str(e)}")
                        continue
                
                # Store info for later use
                self.epochs_shape = epochs_dset.shape
                self.has_encodings = (epoch_type == 'around_evoked')
                
                # For small datasets, load into memory
                if self.epochs_shape[0] < 1000:
                    self.epochs_data = np.array(epochs_dset)
                    if self.has_encodings:
                        self.one_hot_encodings = np.array(ohe_dset)
                else:
                    self.epochs_data = None
            
            # Close the file
            f.close()

        elif re.match(r'^subj0\d_session\d', subject_session_id):
            eeg_file = os.path.join(data_dir, read_from, subject_session_id + '_eeg.fif')
            self.raw = mne.io.read_raw_fif(eeg_file)
            epo, ohe = epoch_around_events(self.raw, before, after, channel_names, resample_freq) if epoch_type == 'around_evoked' else epoch_fixed_lengths(self.raw, fixed_length_duration, channel_names, resample_freq)
            
            # Store as numpy arrays directly
            self.epochs_data = np.array(epo.get_data(), dtype=np.float32)
            self.one_hot_encodings = ohe
            self.epochs_shape = self.epochs_data.shape
            self.has_encodings = (ohe is not None)
            
            # No HDF5 for single subject
            self.group_name = None

        else:
            raise ValueError("subject_session_id must be either 'cross' or should follow the prefix pattern 'subjxx_sessionx'")
        
        self.all_indices = list(range(self.epochs_shape[0]))
        try:
            if len(parts) == 2:
                _, test_pct = parts
                self.train_indices, self.test_indices = train_test_split(
                    self.all_indices, test_size=test_pct / 100, random_state=random_state
                )
                self.val_indices = []  # For consistency
            elif len(parts) == 3:
                _, val_pct, test_pct = parts
                
                self.train_indices, temp = train_test_split(
                    self.all_indices, test_size=(val_pct + test_pct) / 100, random_state=random_state
                )
                
                relative_val_pct = val_pct / (val_pct + test_pct)
                self.val_indices, self.test_indices = train_test_split(
                    temp, test_size=(1 - relative_val_pct), random_state=random_state
                )
        except ValueError as e:
            raise ValueError(
                f"Ensure the dataset split is in the format 'train/val/test' or 'train/test'."
                f"Separately, the fixed length duration may be dividing the data into smaller chunks than expected,"
                f" leaving 0 samples for the test set. Try adjusting the fixed length duration."
            )
        
        self.set_split_type(split_type)

    def set_split_type(self, split_type):
        if split_type == "train":
            self.indices = self.train_indices
        elif split_type == "test":
            self.indices = self.test_indices
        elif split_type == "val":
            self.indices = self.val_indices
        else:
            self.indices = self.all_indices

    # This function updates the group name so that the __getitem__ method reads from the new group name (ground-truth or super-resolution)
    def read_from_super_resolution(self, identifier=''):
        if self.read_from == "super-resolution":
            return
        if not identifier:
            print('No super resolution identifier provided')
            return
        group_parts = self.group_name.split('/')
        self.read_from = "super-resolution"
        group_parts[1] = self.read_from
        group_parts.append(identifier)
        self.group_name = '/'.join(group_parts)

    # This function updates the group name so that the __getitem__ method reads from the new group name (ground-truth or super-resolution)
    def read_from_ground_truth(self):
        if self.read_from == "ground-truth":
            return
        group_parts = self.group_name.split('/')
        self.read_from = "ground-truth"
        group_parts[1] = self.read_from
        group_parts.pop() # remove the identifier when it was reading from super resolution
        self.group_name = '/'.join(group_parts)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        idx = self.indices[index]
        
        # If data is already in memory, use it directly (fast)
        if self.epochs_data is not None:
            if self.has_encodings and self.one_hot_encodings is not None:
                return self.epochs_data[idx], self.one_hot_encodings[idx]
            else:
                return self.epochs_data[idx]
        
        # Otherwise, load from file (slower but memory efficient)
        with h5py.File(self.h5_path, 'r') as f:
            epoch_data = f[f"{self.group_name}/epochs"][idx]
            
            if self.has_encodings:
                one_hot = f[f"{self.group_name}/one-hot-encodings"][idx]
                return epoch_data, one_hot
            else:
                return epoch_data

    def push_super_resolution_to_dataset(self, item, identifier):
        """
        Push super-resolution epoch data and optional one-hot-encoding to the h5 dataset.
        
        Parameters:
        -----------
        item : tuple
            Tuple containing (epoch_data, one_hot_encoding) where one_hot_encoding can be None
        identifier : str
            Identifier to append to the group name
        """
        # Unpack the item
        epoch_data, one_hot = item
        
        # Compute new group name for super-resolution
        group_parts = self.group_name.split('/')
        group_parts[1] = "super-resolution"
        group_parts.append(identifier)
        sr_group_name = '/'.join(group_parts)
        
        with h5py.File(self.h5_path, 'a') as f:
            # Check if super-resolution group exists
            create_datasets = False
            try:
                group = f[sr_group_name]
                current_size = group['epochs'].shape[0]
            except (KeyError, ValueError):
                # Group doesn't exist, create it
                create_datasets = True
                f.create_group(sr_group_name)
                current_size = 0
            
            # If datasets don't exist, create them with same shape as ground-truth
            if create_datasets:
                # Get shape from the original data
                with h5py.File(self.h5_path, 'r') as f_read:
                    original_group = group_parts.copy()
                    original_group[1] = "ground-truth"
                    original_group = '/'.join(original_group[:len(original_group)-1])  # Remove identifier
                    
                    # Get shapes from original datasets
                    channels, timesteps = f_read[f"{original_group}/epochs"].shape[1:]
                    
                    # Create epochs dataset
                    f.create_dataset(
                        f"{sr_group_name}/epochs",
                        shape=(0, channels, timesteps),
                        maxshape=(None, channels, timesteps),
                        dtype=np.float32
                    )
                    
                    # Create one-hot-encodings dataset if needed
                    if one_hot is not None and f"{original_group}/one-hot-encodings" in f_read:
                        num_categories = f_read[f"{original_group}/one-hot-encodings"].shape[1]
                        f.create_dataset(
                            f"{sr_group_name}/one-hot-encodings",
                            shape=(0, num_categories),
                            maxshape=(None, num_categories),
                            dtype=np.float32
                        )
            
            # Add epoch data
            epochs_dset = f[f"{sr_group_name}/epochs"]
            new_size = current_size + epoch_data.shape[0]
            epochs_dset.resize(new_size, axis=0)
            epochs_dset[current_size:new_size] = epoch_data
            
            # Add one-hot encoding if provided
            if one_hot is not None and f"{sr_group_name}/one-hot-encodings" in f:
                ohe_dset = f[f"{sr_group_name}/one-hot-encodings"]
                ohe_dset.resize(new_size, axis=0)
                ohe_dset[current_size:new_size] = one_hot
        
    def has_super_res(self, identifier):
        """
        Check if super-resolution data with the specified identifier exists.
        
        Parameters:
        -----------
        identifier : str
            Identifier appended to the super-resolution group name
        
        Returns:
        --------
        bool
            True if the super-resolution dataset exists, False otherwise
        """
        # Compute the super-resolution group name
        group_parts = self.group_name.split('/')
        
        # If currently reading from super-resolution, handle it
        if group_parts[1] == "super-resolution":
            # Make a copy to avoid altering the current group_name
            group_parts = group_parts.copy()
            # If the current group has an identifier, remove it
            if len(group_parts) > 7:  # Standard group has 7 parts
                group_parts.pop()
        
        # Set to super-resolution and add the identifier
        group_parts[1] = "super-resolution"
        group_parts.append(identifier)
        sr_group_name = '/'.join(group_parts)
        
        # Check if the group exists in the h5 file
        try:
            with h5py.File(self.h5_path, 'r') as f:
                return sr_group_name in f and 'epochs' in f[sr_group_name]
        except Exception as e:
            print(f"Error checking for super-resolution dataset: {str(e)}")
            return False