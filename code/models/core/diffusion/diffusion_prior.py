from torch import nn
from diffusers.models.embeddings import Timesteps, TimestepEmbedding

class DiffusionPriorUNet(nn.Module):
    def __init__(self, embed_dim=1024, cond_dim=42, hidden_dim=[1024, 512, 256, 128, 64], time_embed_dim=512, act_fn=nn.SiLU, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.time_proj = Timesteps(time_embed_dim, True, 0)
        self.input_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim[0]),
            nn.LayerNorm(hidden_dim[0]),
            act_fn(),
        )
        self.num_layers = len(hidden_dim)
        self.encode_time_embedding = nn.ModuleList([TimestepEmbedding(time_embed_dim, hidden_dim[i]) for i in range(self.num_layers-1)])
        self.encode_cond_embedding = nn.ModuleList([nn.Linear(cond_dim, hidden_dim[i]) for i in range(self.num_layers-1)])
        self.encode_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim[i], hidden_dim[i+1]),
                    nn.LayerNorm(hidden_dim[i+1]),
                    act_fn(),
                    nn.Dropout(dropout),
                ) 
                for i in range(self.num_layers-1)
            ]
        )
        self.decode_time_embedding = nn.ModuleList([TimestepEmbedding(time_embed_dim, hidden_dim[i]) for i in range(self.num_layers-1,0,-1)])
        self.decode_cond_embedding = nn.ModuleList([nn.Linear(cond_dim, hidden_dim[i]) for i in range(self.num_layers-1,0,-1)])
        self.decode_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim[i], hidden_dim[i-1]),
                    nn.LayerNorm(hidden_dim[i-1]),
                    act_fn(),
                    nn.Dropout(dropout),
                ) 
                for i in range(self.num_layers-1,0,-1)
            ]
        )
        self.output_layer = nn.Linear(hidden_dim[0], embed_dim)
        

    def forward(self, x, t, c=None):
        t = self.time_proj(t)
        x = self.input_layer(x)

        hidden_activations = []
        for i in range(self.num_layers-1):
            hidden_activations.append(x)
            t_emb = self.encode_time_embedding[i](t) 
            c_emb = self.encode_cond_embedding[i](c) if c is not None else 0
            x = x + t_emb + c_emb
            x = self.encode_layers[i](x)

        for i in range(self.num_layers-1):
            t_emb = self.decode_time_embedding[i](t)
            c_emb = self.decode_cond_embedding[i](c) if c is not None else 0
            x = x + t_emb + c_emb
            x = self.decode_layers[i](x)
            x += hidden_activations[-1-i]

        x = self.output_layer(x)
        return x