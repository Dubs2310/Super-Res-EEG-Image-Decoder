import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class DiffusionEmbeddingDataset(Dataset):
    def __init__(self, img_pair_df: pd.DataFrame, eeg_embeddings_file, features):
        self.df = img_pair_df.copy()
        self.eeg_embeddings = np.load(eeg_embeddings_file)
        self.image_features = features['image_features']
        self.img_id_to_indices = features['img_id_to_indices']
        
        self.embedding_index_map = {}
        for idx, row in self.df.iterrows():
            key = (row['subject'], row['session'], row['epoch_idx'])
            self.embedding_index_map[key] = idx
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        key = (row['subject'], row['session'], row['epoch_idx'])
        eeg_idx = self.embedding_index_map[key]
        eeg_embedding = torch.tensor(self.eeg_embeddings[eeg_idx], dtype=torch.float32)
        
        img_id = int(row['img_id'])
        indices = self.img_id_to_indices[img_id]
        image_features = self.image_features[indices][0]
        
        return { "c_embedding": eeg_embedding, "h_embedding": image_features }


# from torch.utils.data import Dataset

# class DiffusionEmbeddingDataset(Dataset):
#     def __init__(self, c_embeddings, h_embeddings):
#         self.c_embeddings = c_embeddings
#         self.h_embeddings = h_embeddings

#     def __len__(self):
#         return len(self.c_embeddings)  

#     def __getitem__(self, idx):
#         return {
#             "c_embedding": self.c_embeddings[idx],
#             "h_embedding": self.h_embeddings[idx]
#         }

# class EmbeddingDataset(Dataset):
#     def __init__(self, c_embeddings=None, h_embeddings=None, h_embeds_uncond=None, cond_sampling_rate=0.5):
#         self.c_embeddings = c_embeddings
#         self.h_embeddings = h_embeddings
#         self.N_cond = 0 if self.h_embeddings is None else len(self.h_embeddings)
#         self.h_embeds_uncond = h_embeds_uncond
#         self.N_uncond = 0 if self.h_embeds_uncond is None else len(self.h_embeds_uncond)
#         self.cond_sampling_rate = cond_sampling_rate

#     def __len__(self):
#         return self.N_cond

#     def __getitem__(self, idx):
#         return {
#             "c_embedding": self.c_embeddings[idx],
#             "h_embedding": self.h_embeddings[idx]
#         }