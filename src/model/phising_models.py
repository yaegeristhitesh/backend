import torch
import torch.nn as nn
import numpy as np

class Attention(nn.Module):
    """Attention layer from your notebook"""
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim))
    
    def forward(self, x):
        # x: [B, T, F]
        scores = torch.tensordot(x, self.w, dims=([2],[0]))
        attn = torch.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(x * attn, dim=1)

class PhishModel(nn.Module):
    """Exact same architecture as your notebook"""
    def __init__(self, emb_matrix):
        super().__init__()
        num_emb, emb_dim = emb_matrix.shape
        
        # Embedding layer
        self.embed = nn.Embedding(num_emb, emb_dim, padding_idx=0)
        self.embed.weight.data.copy_(torch.from_numpy(emb_matrix))
        self.embed.weight.requires_grad = False  # Freeze embeddings
        
        # CNN layer
        self.conv = nn.Conv1d(emb_dim, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        # BiLSTM layer
        self.lstm = nn.LSTM(64, 128, bidirectional=True, batch_first=True)
        
        # Attention layer
        self.attn = Attention(256)  # 128 * 2 for bidirectional
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.drop = nn.Dropout(0.05)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        # Embedding
        x = self.embed(x)           # [B, T, E]
        x = x.transpose(1, 2)       # [B, E, T]
        
        # CNN
        x = torch.relu(self.conv(x))
        x = self.pool(x)            # [B, 64, T/2]
        x = x.transpose(1, 2)       # [B, T/2, 64]
        
        # BiLSTM
        out, _ = self.lstm(x)       # [B, T/2, 256]
        
        # Attention
        ctx = self.attn(out)        # [B, 256]
        
        # Classification
        h = torch.relu(self.fc1(ctx))
        h = self.drop(h)
        return torch.sigmoid(self.fc2(h)).squeeze(1)