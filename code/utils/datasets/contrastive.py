import os
import yaml
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from utils.singletons.coco import COCO
from utils.datasets.base import EEGDataset

def load_config():
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    else:
        return {}


CONFIG = load_config()
DEFAULT_IMAGE_DIR = CONFIG.get('coco', {}).get('image_dir', "S:\\PolySecLabProjects\\eeg-image-decoding\\data\\all-joined-1\\coco\\images")

class EEGContrastiveDataset(EEGDataset):
    def __init__(self, img_pair_df: pd.DataFrame, input_channels, sfreq, window_before_event_ms, window_after_event_ms, features, montage=None, eeg_dir=None, epochs_dir=None, img_dir=None, img_preprocess_transform=None):
        super().__init__(img_pair_df, input_channels, sfreq, window_before_event_ms, window_after_event_ms, montage, eeg_dir, epochs_dir)
        self.img_dir = DEFAULT_IMAGE_DIR if not img_dir else img_dir
        self.super_labels = COCO().get_supercategory_labels()
        self.fine_labels = COCO().get_fine_category_labels()
        
        # Extract features and mapping from the features dictionary
        self.image_features = features['image_features']
        self.text_features = features['text_features']
        self.img_id_to_indices = features['img_id_to_indices']
        
        # Set up image preprocessing
        if img_preprocess_transform is not None:
            self.preprocess = img_preprocess_transform
        else:
            # Default preprocessing for CLIP/OpenCLIP
            self.preprocess = transforms.Compose([
                transforms.Resize((224, 224)),  # Standard CLIP input size
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP normalization
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])

    def _get_output(self, epoch, row):
        img_id = int(row['img_id'])  # Ensure img_id is an integer
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        
        # Load and preprocess image
        img_pil = Image.open(img_path).convert("RGB")  # Ensure RGB format
        image = self.preprocess(img_pil)  # Apply standardized preprocessing
        
        super_labels = torch.tensor(row[self.super_labels].values, dtype=torch.float32)
        fine_labels = torch.tensor(row[self.fine_labels].values, dtype=torch.float32)
        
        # Get the indices for this img_id
        if img_id in self.img_id_to_indices:
            indices = self.img_id_to_indices[img_id]
            
            # Get features for all captions of this image
            image_features_for_img = self.image_features[indices]
            text_features_for_img = self.text_features[indices]
            
            # Option 1: Take the first caption's features
            image_features = image_features_for_img[0]
            text_features = text_features_for_img[0]
            
            # Option 2: Average all captions' features (uncomment if preferred)
            # image_features = image_features_for_img.mean(dim=0)
            # text_features = text_features_for_img.mean(dim=0)
            
            # Option 3: Randomly select one caption's features
            # idx = torch.randint(0, len(indices), (1,)).item()
            # image_features = image_features_for_img[idx]
            # text_features = text_features_for_img[idx]
            
        else:
            raise KeyError(f"img_id {img_id} not found in feature mapping")

        return image, image_features, text_features, super_labels, fine_labels