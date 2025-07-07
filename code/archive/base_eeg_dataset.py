import os
import mne
import clip
import torch
import open_clip
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch.nn import functional as F
from utils.coco_singleton import COCO

def crop_epoch(epoch, original_window_before_ms, new_window_before_ms, new_window_after_ms, sfreq):
   original_before_samples = int(original_window_before_ms * sfreq / 1000)
   new_before_samples = int(new_window_before_ms * sfreq / 1000)
   new_after_samples = int(new_window_after_ms * sfreq / 1000)
   start_idx = original_before_samples - new_before_samples
   end_idx = start_idx + new_before_samples + new_after_samples
   cropped_epoch = epoch[:, start_idx:end_idx]
   return cropped_epoch


class BaseEEGDataset(Dataset):
    def __init__(self, eeg_img_df: pd.DataFrame, eeg_montage, eeg_input_channels, resampling_frequency, window_before_event_ms, window_after_event_ms, read_eeg_dir, save_epochs_dir, dataset_split='train'):
        self.df = eeg_img_df
        self.eeg_montage = eeg_montage
        self.eeg_input_channels = eeg_input_channels
        self.resampling_frequency = resampling_frequency
        self.window_before_event_ms = window_before_event_ms
        self.window_after_event_ms = window_after_event_ms
        self.read_eeg_dir = read_eeg_dir
        self.save_epochs_dir = os.path.join(save_epochs_dir, f'{window_after_event_ms + window_after_event_ms}ms-{resampling_frequency}Hz')
        self.dataset_split = dataset_split

        subject_session_df = self.df[['subject', 'session']].drop_duplicates()
        self.eeg_file_paths = [os.path.join(read_eeg_dir, f"subj0{row['subject']}_session{row['session']}_eeg.fif") for _, row in subject_session_df.iterrows()]
        os.makedirs(save_epochs_dir, exist_ok=True)

        montage = mne.channels.make_standard_montage(self.eeg_montage)
        self.eeg_montage_channel_names = list(montage.get_positions()['ch_pos'].keys())
        subject_session_df = self.df[['subject', 'session']].drop_duplicates()
        
        for _, row in subject_session_df.iterrows():
            subject = row['subject']
            session = row['session']
            epoch_file = self._get_epoch_filepath(subject, session)
            eeg_file = os.path.join(read_eeg_dir, f"subj0{subject}_session{session}_eeg.fif")
            raw = mne.io.read_raw_fif(eeg_file, preload=True)
            raw.resample(resampling_frequency)
            events = mne.find_events(raw)
            tmin = -window_before_event_ms / 1000
            tmax = window_after_event_ms / 1000
            epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, preload=True, baseline=None)
            epochs_data = epochs.get_data()
            np.save(epoch_file, epochs_data)
            del raw, epochs, epochs_data

    def _load_eeg_epoch(self, row):
        subject = row['subject']
        session = row['session']
        epoch_idx = row['epoch_idx']
        epoch_file = self._get_epoch_filepath(subject, session)
        epochs_data = np.load(epoch_file)
        epoch = epochs_data[epoch_idx]
        channel_indices = [i for i, ch in enumerate(self.eeg_montage_channel_names) if ch in self.eeg_input_channels]
        epoch = epoch[channel_indices, :]
        return torch.tensor(epoch, dtype=torch.float32)

    def _get_epoch_filepath(self, subject, session):
        return os.path.join(self.save_epochs_dir, f'subj0{subject}_session{session}_epochs.npy')
    
    def _get_output(self, row):
        raise NotImplementedError

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        epoch = self._load_eeg_epoch(row)
        output = self._get_output(row)
        return epoch, output


class SuperResEEGDataset(BaseEEGDataset):
    def __init__(self, eeg_img_df: pd.DataFrame, eeg_montage, eeg_input_channels, eeg_output_channels, resampling_frequency, window_before_event_ms, window_after_event_ms, read_eeg_dir, save_epochs_dir, dataset_split='train'):
        self.eeg_output_channels = eeg_output_channels
        super().__init__(eeg_img_df, eeg_montage, eeg_input_channels, resampling_frequency, window_before_event_ms, window_after_event_ms, read_eeg_dir, save_epochs_dir, dataset_split)
    
    def _get_output(self, row):
        subject = row['subject']
        session = row['session']
        epoch_idx = row['epoch_idx']
        epoch_file = self._get_epoch_filepath(subject, session)
        epochs_data = np.load(epoch_file)
        epoch = epochs_data[epoch_idx]
        channel_indices = [i for i, ch in enumerate(self.channel_names) if ch in self.eeg_output_channels]
        output_epoch = epoch[channel_indices, :]
        return torch.tensor(output_epoch, dtype=torch.float32)


class EEGImageDataset(BaseEEGDataset):
    def __init__(self, eeg_img_df: pd.DataFrame, eeg_montage, eeg_input_channels, resampling_frequency, window_before_event_ms, window_after_event_ms, read_eeg_dir, save_epochs_dir, img_dir, dataset_split='train', latent_id='random_42'):
        self.img_dir = img_dir
        model_type = 'ViT-H-14'
        pretrained = 'laion2b_s32b_b79k'
        precision = 'fp32'
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.vlmodel, self.preprocess_train, self.feature_extractor = open_clip.create_model_and_transforms(model_type=model_type, pretrained=pretrained, precision=precision, device=self.device)
        super().__init__(eeg_img_df, eeg_montage, eeg_input_channels, resampling_frequency, window_before_event_ms, window_after_event_ms, read_eeg_dir, save_epochs_dir, dataset_split)
        
        unique_img_ids = self.df['img_id'].unique()
        self.captions = []
        self.images = []
        
        for img_id in unique_img_ids:
            captions = COCO()[img_id]['captions'] 
            img_path = os.path.join(self.img_dir, f"{img_id:012d}.jpg")
            self.captions.extend(captions)
            self.images.extend([img_path] * len(captions))

        features_filename = os.path.join(f'{model_type}_features_{dataset_split}_{latent_id}.pt')

        if os.path.exists(features_filename):
            saved_features = torch.load(features_filename)
            self.text_features = saved_features['text_features']
            self.image_features = saved_features['image_features']
        else:
            self.text_features = self._encode_text_captions(self.captions)
            self.image_features = self._encode_images(self.images)
            torch.save({ 'text_features': self.text_features.cpu(), 'image_features': self.image_features.cpu() }, features_filename)

    # Adapted from EEG Image Decode
    def _encode_text_captions(self, texts):
        text_inputs = torch.cat([clip.tokenize(t) for t in texts]).to(self.device)
        with torch.no_grad():
            text_features = self.vlmodel.encode_text(text_inputs)
        text_features = F.normalize(text_features, dim=-1).detach()
        return text_features
    
    # Adapted from EEG Image Decode
    def _encode_images(self, images, batch_size=20):
        batch_size = 20 
        image_features_list = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            image_inputs = torch.stack([self.preprocess_train(Image.open(img).convert("RGB")) for img in batch_images]).to(self.device)
            with torch.no_grad():
                batch_image_features = self.vlmodel.encode_image(image_inputs)
                batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)
            image_features_list.append(batch_image_features)
        image_features = torch.cat(image_features_list, dim=0)        
        return image_features

    def _get_output(self, row):
        img_id = row['img_id']
        img_path = os.path.join(self.img_dir, f"{img_id:012d}.jpg")
        img = Image.open(img_path)
        img_array = np.array(img)
        return torch.tensor(img_array, dtype=torch.float32), self.image_features, self.text_features


class MultilabelEEGDataset(BaseEEGDataset):
    def __init__(self, eeg_img_df: pd.DataFrame, eeg_montage, eeg_input_channels, resampling_frequency, window_before_event_ms, window_after_event_ms, read_eeg_dir, save_epochs_dir, dataset_split='train'):
        super().__init__(eeg_img_df, eeg_montage, eeg_input_channels, resampling_frequency, window_before_event_ms, window_after_event_ms, read_eeg_dir, save_epochs_dir, dataset_split)
        self.sc_columns = [col for col in self.df.columns if '(sc)' in col]
        self.fc_columns = [col for col in self.df.columns if '(fc)' in col]
        self.all_label_columns = self.sc_columns + self.fc_columns
    
    def _get_output(self, row):
        labels = row[self.all_label_columns].values
        return torch.tensor(labels, dtype=torch.float32)
    

class EEGPredictionDataset(BaseEEGDataset):
    def __init__(self, eeg_img_df: pd.DataFrame, eeg_montage, eeg_input_channels, resampling_frequency, window_before_event_ms, window_after_event_ms, read_eeg_dir, save_epochs_dir, model, dataset_split='train'):
        self.model = model
        self.model.eval()
        super().__init__(eeg_img_df, eeg_montage, eeg_input_channels, resampling_frequency, window_before_event_ms, window_after_event_ms, read_eeg_dir, save_epochs_dir, dataset_split)
    
    def _get_output(self, row):
        subject = row['subject']
        session = row['session']
        epoch_idx = row['epoch_idx']
        epoch_file = self._get_epoch_filepath(subject, session)
        epochs_data = np.load(epoch_file)
        epoch = epochs_data[epoch_idx]
        channel_indices = [i for i, ch in enumerate(self.channel_names) if ch in self.eeg_input_channels]
        input_epoch = epoch[channel_indices, :]
        input_tensor = torch.tensor(input_epoch, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)
        return output.squeeze(0)