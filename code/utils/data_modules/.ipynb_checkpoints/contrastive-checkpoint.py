import torch
from utils.data_modules.base import EEGDataModule
from utils.datasets.contrastive import EEGContrastiveDataset

# eeg_input_params = {
#     'eeg_montage': 'standard_1020',
#     'eeg_input_channels': ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'],
#     'resampling_frequency': 250,
#     'window_before_event_ms': 50,
#     'window_after_event_ms': 600,
#     'read_eeg_dir': "S:\\PolySecLabProjects\\eeg-image-decoding\\data\\all-joined-1\\eeg\\preprocessed",
#     'save_epochs_dir': "S:\\PolySecLabProjects\\eeg-image-decoding\\data\\all-joined-1\\eeg\\epochs"
# }

# img_dir = "S:\\PolySecLabProjects\\eeg-image-decoding\\data\\all-joined-1\\coco\\images"

class EEGContrastiveDataModule(EEGDataModule):
    def __init__(self, input_channels, sfreq, window_before_event_ms, window_after_event_ms, montage=None, eeg_dir=None, epochs_dir=None, img_dir=None, subject=None, session=None, batch_size=32, num_workers=4, val_split=0.1, test='default'):
        self.img_dir = img_dir
        self.base_datamodule_params = {
            'dataset_class': EEGContrastiveDataset,
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
            'val_split': val_split,
            'test': test
        }
        super().__init__(**self.base_datamodule_params)

    def get_dataset_output_params(self, df, split='train'):
        if split == 'all':
            # Get both train and test features
            _, train_img_feat, train_text_feat, train_img_to_indices = self.coco.get_train_set()
            _, test_img_feat, test_text_feat, test_img_to_indices = self.coco.get_test_set()
            
            # Concatenate features
            concatenated_img_feat = torch.cat([train_img_feat, test_img_feat], dim=0)
            concatenated_text_feat = torch.cat([train_text_feat, test_text_feat], dim=0)
            
            # Merge img_id_to_indices dictionaries
            concatenated_img_to_indices = {**train_img_to_indices, **test_img_to_indices}
            
            features = { 
                'image_features': concatenated_img_feat,
                'text_features': concatenated_text_feat,
                'img_id_to_indices': concatenated_img_to_indices
            }
        elif split in ['train', 'val']:
            _, train_img_feat, train_text_feat, train_img_to_indices = self.coco.get_train_set()
            features = { 
                'image_features': train_img_feat,
                'text_features': train_text_feat,
                'img_id_to_indices': train_img_to_indices
            }
        else:
            _, test_img_feat, test_text_feat, test_img_to_indices = self.coco.get_test_set()
            features = { 
                'image_features': test_img_feat,
                'text_features': test_text_feat,
                'img_id_to_indices': test_img_to_indices
            }
        
        return {
            'img_dir': self.img_dir,
            'features': features
        }
    

    def get_output_sample_info(self, output_sample):
        return {
            'image_shape': tuple(output_sample[0].shape),
            'image_features_shape': tuple(output_sample[1].shape),
            'text_features_shape': tuple(output_sample[2].shape),
            'super_labels_shape': len(output_sample[3]),
            'fine_labels_shape': len(output_sample[4])
        }