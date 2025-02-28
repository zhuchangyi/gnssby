import torch
import torch.nn as nn

import torch.nn.functional as F
import math
from sklearn.cluster import KMeans

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        return self.w_2(self.dropout(self.leaky_relu(self.w_1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.layernorm1(x)
        x = x + self.dropout1(self.self_attn(x2, x2, x2, attn_mask=mask)[0])
        x2 = self.layernorm2(x)
        x = x + self.dropout2(self.ffn(x2))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, d_ff, num_layers, dropout=0):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class GNSSClusterer(nn.Module):
    def __init__(self, d_model=64, nhead=4, d_ff=64, num_layers=3, num_satellites=81, num_clusters=5, dropout=0):
        super(GNSSClusterer, self).__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(15, 32),
            nn.ReLU(),
            nn.Linear(32, d_model)
        )

        self.transformer_encoder = TransformerEncoder(d_model, nhead, d_ff, num_layers)
        self.num_clusters = num_clusters

    def forward(self, x, apply_kmeans=False):
        batch_size = x.size(0)
        x = self.input_proj(x)
        x = x.transpose(0, 1)  # (num_satellites, batch_size, d_model)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1).contiguous().view(batch_size, -1)  # (batch_size, num_satellites * d_model)

        if apply_kmeans:
            if x.shape[0] >= self.num_clusters:
                kmeans = KMeans(n_clusters=self.num_clusters, n_init=10, random_state=0)
                cluster_labels = kmeans.fit_predict(x.detach().cpu().numpy())
            else:
                raise ValueError(f"Number of samples ({x.shape[0]}) should be >= number of clusters ({self.num_clusters}).")
            return cluster_labels

        return x

