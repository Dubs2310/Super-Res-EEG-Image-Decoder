import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils.singletons.coco import COCO
from sklearn.model_selection import train_test_split


class DiffusionEmbeddingDataModule(pl.LightningDataModule):
    def __init__(self, eeg_embeddings_file, subject=None, session=None, batch_size=32, num_workers=4, val_split=0.1, test='default'):
        super().__init__()
        self.eeg_embeddings_file = eeg_embeddings_file
        self.subject = subject
        self.session = session
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test = test
        self.coco = COCO()

    def _filter_dataframe(self, df):
        if self.subject and self.session:
            return df[(df['subject'] == self.subject) & (df['session'] == self.session)]
        elif self.subject and not self.session:
            return df[df['subject'] == self.subject]
        else:
            return df
    
    def _get_features(self, split):
        if split == 'all':
            _, train_img_feat, _, train_img_to_indices = self.coco.get_train_set()
            _, test_img_feat, _, test_img_to_indices = self.coco.get_test_set()
            concatenated_img_feat = torch.cat([train_img_feat, test_img_feat], dim=0)
            concatenated_img_to_indices = {**train_img_to_indices, **test_img_to_indices}
            return { 
                'image_features': concatenated_img_feat,
                'img_id_to_indices': concatenated_img_to_indices
            }
        elif split in ['train', 'val']:
            _, train_img_feat, _, train_img_to_indices = self.coco.get_train_set()
            return { 
                'image_features': train_img_feat,
                'img_id_to_indices': train_img_to_indices
            }
        else:
            _, test_img_feat, _, test_img_to_indices = self.coco.get_test_set()
            return { 
                'image_features': test_img_feat,
                'img_id_to_indices': test_img_to_indices
            }

    def setup(self, stage=None):
        if hasattr(self, 'train_dataset') and hasattr(self, 'val_dataset') and hasattr(self, 'test_dataset'):
            return
            
        if self.test == 'All':
            all_df = self.coco.get_global_eeg_image_df()
            test_df = self._filter_dataframe(all_df)
            
            train_df, _, _, _ = self.coco.get_train_set()
            train_df = self._filter_dataframe(train_df)
            
            if len(train_df) == 0:
                train_df = test_df.iloc[:0]
                val_df = test_df.iloc[:0]
            else:
                train_df, val_df = train_test_split(train_df, test_size=self.val_split, random_state=42)
        else:
            train_df, _, _, _ = self.coco.get_train_set()
            test_df, _, _, _ = self.coco.get_test_set()
            train_df = self._filter_dataframe(train_df)
            test_df = self._filter_dataframe(test_df)
            
            train_df, val_df = train_test_split(train_df, test_size=self.val_split, random_state=42)
        
        train_features = self._get_features('train')
        val_features = self._get_features('val')
        test_features = self._get_features('all' if self.test == 'All' else 'test')
        
        self.train_dataset = DiffusionEmbeddingDataset(train_df, self.eeg_embeddings_file, train_features)
        self.val_dataset = DiffusionEmbeddingDataset(val_df, self.eeg_embeddings_file, val_features)
        self.test_dataset = DiffusionEmbeddingDataset(test_df, self.eeg_embeddings_file, test_features)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)