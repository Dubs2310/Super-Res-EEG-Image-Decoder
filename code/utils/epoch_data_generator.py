import mne
import os
import pandas as pd
import numpy as np
from keras.api.utils import Sequence
from enum import Enum

class OutputType(Enum):
    ALL_CHANNELS = "all_channels"  # Use all available channels as output
    MIRROR_INPUT = "mirror_input"   # Output matches input channels (for autoencoders)
    IMAGE_LABEL = "image_label"     # Output is image label
    IMAGE = "image"                 # Output is image data
    SUPER_RESOLUTION = "super_resolution"  # Output is high-res channels from low-res input
    

class EpochDataGenerator(Sequence):
    @staticmethod
    def data_dir():
        return os.path.join(os.path.dirname(__file__), "..", "..", "data", "things-eeg")

    def __init__(self, subject_epoch_coordinates, lr_channel_names=None, hr_channel_names=None, 
                 output_type=OutputType.IMAGE_LABEL):
        self.subject_epoch_coordinates = subject_epoch_coordinates
        self.lr_channel_names = lr_channel_names
        self.hr_channel_names = hr_channel_names
        self.output_type = output_type
        item = self[0]
        self.in_sh = item[0].shape[-2:]
        self.out_sh = item[1].shape[-2:]

    def __len__(self):
        return len(self.subject_epoch_coordinates)
    
    def get_input_shape(self):
        return len(self), *self.in_sh
    
    def get_output_shape(self):
        return len(self), *self.out_sh
    
    def set_output_type(self, output_type):
        self.output_type = output_type
        self.out_sh = self[0][1].shape[-2:] # update output shape based on the new output type

    def __getitem__(self, index):
        subject, epoch_index = self.subject_epoch_coordinates[index]
        
        # Load full epoch data
        full_epoch = mne.read_epochs(os.path.join(self.data_dir(), subject, "eeg", 
                                              f"{subject}_task-rsvp_eeg_epochs.fif"))[epoch_index]
        
        # Handle different input/output configurations
        if self.output_type == OutputType.SUPER_RESOLUTION:
            if self.lr_channel_names is None or self.hr_channel_names is None:
                raise ValueError("lr_channel_names and hr_channel_names must be provided for SUPER_RESOLUTION mode")
            
            # Get low-resolution input
            lr_epoch = full_epoch.copy().pick(self.lr_channel_names)
            lr_data = np.squeeze(lr_epoch.get_data())
            
            # Get high-resolution output (target)
            hr_epoch = full_epoch.copy().pick(self.hr_channel_names)
            hr_data = np.squeeze(hr_epoch.get_data())
            
            return lr_data, hr_data
            
        elif self.lr_channel_names is not None:
            # Use only specified lr channels for input
            epoch = full_epoch.copy().pick(self.lr_channel_names)
            input_data = np.squeeze(epoch.get_data())
        else:
            # Use all available channels
            input_data = np.squeeze(full_epoch.get_data())
        
        # For all other output types, handle as before
        if self.output_type == OutputType.ALL_CHANNELS:
            out = np.squeeze(full_epoch.get_data())
            return input_data, out

        elif self.output_type == OutputType.MIRROR_INPUT:
            return input_data, input_data
        
        elif self.output_type == OutputType.IMAGE_LABEL:
            label = pd.read_csv(os.path.join(self.data_dir(), subject, "eeg", 
                                         f"{subject}_task-rsvp_events.tsv"), 
                            sep="\t")["object"].to_numpy()[epoch_index]
            return input_data, label
        
        elif self.output_type == OutputType.IMAGE:
            # Assuming there's image data to return
            # This would need to be implemented based on how images are stored
            raise NotImplementedError("IMAGE output type not yet implemented")
        
        return input_data, None