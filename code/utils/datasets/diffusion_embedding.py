from torch.utils.data import Dataset

class DiffusionEmbeddingDataset(Dataset):
    def __init__(self, c_embeddings, h_embeddings):
        self.c_embeddings = c_embeddings
        self.h_embeddings = h_embeddings

    def __len__(self):
        return len(self.c_embeddings)  

    def __getitem__(self, idx):
        return {
            "c_embedding": self.c_embeddings[idx],
            "h_embedding": self.h_embeddings[idx]
        }

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