import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class CNNSpeakerModel(nn.Module):
    """CNN-based speaker recognition model for voice biometrics"""
    
    def __init__(self, input_dim: int = 13, embedding_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # CNN layers for MFCC feature extraction
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, embedding_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, 1, n_mfcc, time_frames)
        Returns:
            Speaker embedding of shape (batch_size, embedding_dim)
        """
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # L2 normalize for cosine similarity
        x = F.normalize(x, p=2, dim=1)
        
        return x

class SpeakerEmbeddingExtractor:
    """Extract speaker embeddings from MFCC features"""
    
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNNSpeakerModel().to(self.device)
        
        if model_path:
            self._load_model(model_path)
        else:
            # Initialize with random weights for demo
            logger.info("Using randomly initialized speaker model for demo")
        
        self.model.eval()
    
    def _load_model(self, model_path: str):
        """Load pre-trained model weights"""
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded speaker model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load speaker model: {e}")
            raise
    
    def extract_embedding(self, mfcc_features: np.ndarray) -> np.ndarray:
        """
        Extract speaker embedding from MFCC features
        
        Args:
            mfcc_features: MFCC features of shape (n_mfcc, time_frames)
        
        Returns:
            Speaker embedding vector
        """
        try:
            # Prepare input tensor
            if len(mfcc_features.shape) == 2:
                # Add batch and channel dimensions
                mfcc_tensor = torch.FloatTensor(mfcc_features).unsqueeze(0).unsqueeze(0)
            else:
                raise ValueError(f"Expected 2D MFCC features, got shape {mfcc_features.shape}")
            
            mfcc_tensor = mfcc_tensor.to(self.device)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model(mfcc_tensor)
                embedding = embedding.cpu().numpy().flatten()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            # Return zero embedding on failure
            return np.zeros(self.model.embedding_dim)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            return 0.0
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model_type": "CNN Speaker Recognition",
            "embedding_dim": self.model.embedding_dim,
            "input_dim": self.model.input_dim,
            "device": str(self.device),
            "parameters": sum(p.numel() for p in self.model.parameters())
        }