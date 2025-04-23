import mne
import os
import pandas as pd
from keras.api.utils import Sequence
from enum import Enum

class OutputType(Enum):
    ALL_CHANNELS = "all_channels"
    MIRROR_INPUT = "mirror_input"
    IMAGE_LABEL = "image_label"
    IMAGE = "image"
    

class EpochDataGenerator(Sequence):
    @staticmethod
    def data_dir():
        return os.path.join("..", "..", "data", "things-eeg")

    def __init__(self, subject_epoch_coordinates, channel_names=None, output_type=OutputType.IMAGE_LABEL):
        self.subject_epoch_coordinates = subject_epoch_coordinates
        self.channel_names = channel_names
        self.output_type = output_type
        
        item = self[0]
        self.in_sh = item[0].shape[1:]
        self.out_sh = item[1].shape[1:]

    def __len__(self):
        return len(self.subject_epoch_coordinates)
    
    def get_input_shape(self):
        return len(self), *self.in_sh
    
    def get_output_shape(self):
        return len(self), *self.out_sh
    
    def set_output_type(self, output_type):
        self.output_type = output_type
        self.out_sh = self[0][1].shape[1:] # update output shape based on the new output type

    def __getitem__(self, index):
        subject, epoch_index = self.subject_epoch_coordinates[index]

        epoch = mne.read_epochs(os.path.join(self.data_dir(), subject, "eeg", f"{subject}_task-rsvp_eeg_epochs.fif"))[epoch_index]
        
        if self.channel_names is not None:
            epoch = epoch.pick(self.channel_names)

        epoch = epoch.get_data()

        # For eeg super resolution
        if self.output_type == OutputType.ALL_CHANNELS:
            out = mne.read_epochs(os.path.join(self.data_dir(), subject, "eeg", f"{subject}_task-rsvp_eeg_epochs.fif"))[epoch_index].get_data()
            return epoch, out

        # For self-supervised learning of EEGs masked auto encoders (MAEs)
        if self.output_type == OutputType.MIRROR_INPUT:
            return epoch, epoch
        
        # For image classification
        # if self.output_type == OutputType.IMAGE_LABEL:
        label = pd.read_csv(os.path.join(self.data_dir(), subject, "eeg", f"{subject}_task-rsvp_events.tsv"), sep="\t")["object"].to_numpy()[epoch_index]
        return epoch, label