import os
import h5py
import numpy as np
from torch.utils.data import Dataset
# from keras.api.utils import Sequence
from sklearn.model_selection import train_test_split

class HDF5DataSplitGenerator(Dataset):
    def __init__(self, h5_file_path=None, dataset_type="train", test_size=0.2, random_state=97, event_mode="fixed_length_event", event_duration=8, lr_channel_names=None, hr_channel_names=None, batch_size=32):
        """
        Initialize the data generator

        Parameters:
        --------------
        h5_file_path: str or None
            Path to h5 file. If None, defaults to 'comprehensive_dataset.h5' in the data directory

        dataset_type: str
            Type of dataset
                Options: 'train' | 'test'

        test_size: float
            Test split percentage
        
        random_state: int
            Random state for train/test split
        
        event_mode: str
            Mode of epoch event
                Options: 'fixed_length_event' | 'evoked_event'

        event_duration: int
            Fixed duration of epoch event if event_mode = 'fixed_time_length'
                Options: 60 | 30 | 10
        
        lr_channel_names: list or None
            List of low-resolution channel names to use (if None, use all channels)

        hr_channel_names: list or None
            List of high-resolution channel names to use (if output type is super-resolution)

        batch_size: int
            Batch size for training
        """
        
        if event_mode not in ['fixed_length_event', 'evoked_event']:
            raise ValueError("event_mode must be one of ['fixed_length_event', 'evoked_event']")
        
        if event_duration not in [8, 4, 1]:
            raise ValueError("event_duration must be one of [8, 4, 1]")
        
        if dataset_type not in ['train', 'test']:
            raise ValueError("dataset_type must be one of ['train', 'test']")
        
        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")
        
        # TODO: lr_channel_names must be less than or equal to hr_channel_names

        self.event_mode = event_mode
        self.batch_size = batch_size
        
        # Default h5 file path if not provided
        self.h5_file_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "all-joined-1", "comprehensive_dataset.h5") if h5_file_path is None else h5_file_path
        
        # Generate keys for the h5 file
        self.epochs_key = f"all_{f'{event_duration}s' if self.event_mode == 'fixed_length_event' else 'evoked_event'}_epochs" 
        self.metadata_key = f"{self.epochs_key}_metadata"
        
        # Get dataset information without keeping the dataset open
        with h5py.File(self.h5_file_path, 'r') as f:
            self.num_epochs = f[self.epochs_key].shape[0]
            self.ch_names = [ch.decode('utf-8') for ch in f['ch_names'][()]]
        
        # Split indices for train/test sets
        all_indices = np.arange(self.num_epochs)
        self.train_indices, self.test_indices = train_test_split(
            all_indices, test_size=test_size, random_state=random_state
        )
        
        # Use appropriate indices based on dataset_type
        self.indices = self.train_indices if dataset_type == "train" else self.test_indices
        
        # Get channel indices for low and high resolution
        self.lr_indices = None
        if lr_channel_names is not None:
            self.lr_indices = [self.ch_names.index(ch) for ch in lr_channel_names if ch in self.ch_names]
        
        self.hr_indices = None
        if hr_channel_names is not None:
            self.hr_indices = [self.ch_names.index(ch) for ch in hr_channel_names if ch in self.ch_names]
    
    def __len__(self):
        """Return the number of batches"""
        return int(np.ceil(len(self.indices) / self.batch_size))
    
    def __getitem__(self, batch_idx):
        """Get a batch of data"""
        # Check if batch_idx is valid
        if batch_idx >= len(self):
            raise IndexError(f"Index {batch_idx} out of bounds for {self.dataset_type} dataset with {len(self)} batches")

        # Calculate indices for this batch
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.indices))
        batch_indices = self.indices[start_idx:end_idx]
        
        # Open the h5 file and get the batch data
        with h5py.File(self.h5_file_path, 'r') as f:
            if len(batch_indices) == 1:
                # Special case for single-item batch
                batch_data = np.expand_dims(f[self.epochs_key][batch_indices[0]], axis=0)
                batch_metadata = np.array([f[self.metadata_key][batch_indices[0]]])
            else:
                # Multi-item batch - h5py requires indices to be in ascending order
                # Sort indices for h5py access
                sorted_indices = np.sort(batch_indices)
                # Get data using sorted indices
                sorted_batch_data = f[self.epochs_key][sorted_indices]
                sorted_batch_metadata = f[self.metadata_key][sorted_indices]
                
                # Now rearrange to original order if needed
                if not np.array_equal(sorted_indices, batch_indices):
                    # Create mapping from sorted to original positions
                    idx_map = np.zeros_like(batch_indices)
                    for i, idx in enumerate(batch_indices):
                        idx_map[i] = np.where(sorted_indices == idx)[0][0]
                    
                    # Rearrange data back to original order
                    batch_data = sorted_batch_data[idx_map]
                    batch_metadata = sorted_batch_metadata[idx_map]
                else:
                    # Already in correct order
                    batch_data = sorted_batch_data
                    batch_metadata = sorted_batch_metadata
        
        # Filter channels if needed
        lr_batch_data = batch_data[:, self.lr_indices] if self.lr_indices is not None else batch_data
        hr_batch_data = batch_data[:, self.hr_indices] if self.hr_indices is not None else batch_data
        
        return {
            "lo_res_epoch_batch" : lr_batch_data, 
            "hi_res_epoch_batch" : hr_batch_data, 
            "epoch_batch_metadata" : batch_metadata
        }
    
    def get_item_at_index(self, index):
        """Get a single epoch at the specified index (not a batch)"""
        if index >= len(self.indices):
            raise IndexError(f"Index {index} out of bounds for dataset of size {len(self.indices)}")
        
        actual_index = self.indices[index]
        
        with h5py.File(self.h5_file_path, 'r') as f:
            # Get the epoch data
            epoch_data = f[self.epochs_key][actual_index]
            metadata = f[self.metadata_key][actual_index]
        
        # Filter channels if needed
        lr_data = epoch_data[self.lr_indices] if self.lr_indices is not None else epoch_data
        hr_data = epoch_data[self.hr_indices] if self.hr_indices is not None else epoch_data
        
        out = {
            "lo_res_epoch_item" : lr_data, 
            "hi_res_epoch_item" : hr_data, 
            "epoch_item_metadata" : metadata
            # 'subject': metadata[0],
            # 'session': metadata[1],
            # 'sample_number': metadata[2],
        }

        return out

        # if self.event_mode == 'evoked_event':

    def get_data_shape(self):
        """Return the shape of the dataset"""
        with h5py.File(self.h5_file_path, 'r') as f:
            epoch_shape = f[self.epochs_key].shape[1:]
            
        lr_shape = (len(self.lr_indices),) + epoch_shape[1:] if self.lr_indices is not None else epoch_shape
        hr_shape = (len(self.hr_indices),) + epoch_shape[1:] if self.hr_indices is not None else epoch_shape
            
        return {
            "lo_res_shape": lr_shape,
            "hi_res_shape": hr_shape
        }
    
    def get_channel_names(self):
        """Return the channel names used"""
        lr_names = [self.ch_names[i] for i in self.lr_indices] if self.lr_indices is not None else self.ch_names
        hr_names = [self.ch_names[i] for i in self.hr_indices] if self.hr_indices is not None else self.ch_names
        
        return {
            "lo_res_channels": lr_names,
            "hi_res_channels": hr_names
        }