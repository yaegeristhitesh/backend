import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class VoiceCNN(nn.Module):
    """
    CNN-based speaker recognition model.
    Based on: "CNN based speaker recognition in language and text-independent small scale system"
    Enhanced for phishing detection.
    """
    
    def __init__(self, num_speakers: int = 50, feature_dim: int = 52):
        """
        Args:
            num_speakers: Number of speakers to classify
            feature_dim: Input feature dimension (MFCC 13 + delta + delta-delta + filterbank)
        """
        super().__init__()
        
        # Convolutional layers (as per paper architecture)
        # Layer 1: 52 kernels, window size 13
        self.conv1 = nn.Conv1d(
            in_channels=1,  # Single channel for MFCC
            out_channels=52,
            kernel_size=13,
            padding=6  # Same padding
        )
        
        # Layer 2: 52 kernels, window size 7
        self.conv2 = nn.Conv1d(
            in_channels=52,
            out_channels=52,
            kernel_size=7,
            padding=3
        )
        
        # Layer 3: 13 kernels, window size 3
        self.conv3 = nn.Conv1d(
            in_channels=52,
            out_channels=13,
            kernel_size=3,
            padding=1
        )
        
        # Attention mechanism for important frame selection
        self.attention = nn.MultiheadAttention(
            embed_dim=13,
            num_heads=1,
            batch_first=True
        )
        
        # Scam detection branch
        self.scam_detector = nn.Sequential(
            nn.Linear(13 * 50, 256),  # Flattened features
            nn.Tanh(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
        )
        
        # Speaker embedding layer
        self.speaker_embedding = nn.Linear(64, 256)
        
        # Classification heads
        self.speaker_classifier = nn.Linear(256, num_speakers)
        self.scam_classifier = nn.Linear(256, 2)  # Binary: scam vs legitimate
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(52)
        self.bn2 = nn.BatchNorm1d(52)
        self.bn3 = nn.BatchNorm1d(13)
        
        # Dropout layers
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 1, features, frames)
            return_embedding: If True, return embedding only
            
        Returns:
            If return_embedding: speaker embedding
            Else: (speaker_logits, scam_logits)
        """
        # Input shape: (batch, 1, features, frames)
        batch_size = x.shape[0]
        
        # Reshape if needed
        if len(x.shape) == 4:
            x = x.squeeze(1)  # Remove channel dim if present
        
        # Conv layer 1 with tanh activation (as per paper)
        x = torch.tanh(self.conv1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        
        # Conv layer 2
        x = torch.tanh(self.conv2(x))
        x = self.bn2(x)
        x = self.dropout(x)
        
        # Conv layer 3
        x = torch.tanh(self.conv3(x))
        x = self.bn3(x)
        x = self.dropout(x)
        
        # Attention across frames
        # x shape: (batch, features=13, frames)
        x = x.permute(0, 2, 1)  # (batch, frames, features)
        
        # Apply attention
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.mean(dim=1)  # Average over frames
        
        # Flatten for scam detection
        x = x.view(batch_size, -1)
        
        # Scam detection features
        x = self.scam_detector(x)
        
        # Speaker embedding
        embedding = self.speaker_embedding(x)
        
        if return_embedding:
            return embedding
        
        # Speaker classification
        speaker_logits = self.speaker_classifier(embedding)
        
        # Scam classification
        scam_logits = self.scam_classifier(embedding)
        
        return speaker_logits, scam_logits
    
    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding from input features"""
        return self.forward(x, return_embedding=True)
    
    def predict_scam(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict if audio is from a scammer"""
        speaker_logits, scam_logits = self.forward(x)
        scam_probs = F.softmax(scam_logits, dim=-1)
        return scam_probs[:, 1]  # Probability of being scam