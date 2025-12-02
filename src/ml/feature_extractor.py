import numpy as np
import librosa
from typing import Tuple, Optional

class MFCCFeatureExtractor:
    """
    MFCC feature extractor following the research paper specifications.
    Extracts 13 MFCC coefficients + delta + delta-delta + filterbank energies = 52 features
    """
    
    def __init__(self, sample_rate: int = 16000, n_mfcc: int = 13):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.frame_length = int(0.032 * sample_rate)  # 32ms as per paper
        self.hop_length = int(0.016 * sample_rate)    # 16ms stride as per paper
    
    def extract(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> np.ndarray:
        """
        Extract MFCC features from audio.
        
        Args:
            audio: Audio signal as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            MFCC features (n_frames, 52 features)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        # Ensure audio is float32
        audio = audio.astype(np.float32)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
            n_mels=26  # As per common practice
        )
        
        # Extract delta features
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        # Extract filterbank energies
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
            n_mels=13  # 13 filterbanks as per paper
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Stack all features: 13 MFCC + 13 delta + 13 delta-delta + 13 filterbank = 52
        features = np.vstack([
            mfcc,
            delta_mfcc,
            delta2_mfcc,
            log_mel
        ])
        
        # Transpose to (n_frames, n_features)
        features = features.T
        
        # Normalize features
        features = self._normalize_features(features)
        
        return features
    
    def extract_with_context(self, audio: np.ndarray, sample_rate: int, 
                           context_frames: int = 40) -> np.ndarray:
        """
        Extract features with context frames (as mentioned in paper for text-dependent).
        Stack features from multiple frames.
        """
        features = self.extract(audio, sample_rate)
        
        # Pad features
        padded = np.pad(
            features,
            ((context_frames//2, context_frames//2), (0, 0)),
            mode='edge'
        )
        
        # Create stacked features
        stacked_features = []
        for i in range(len(features)):
            context = padded[i:i+context_frames].flatten()
            stacked_features.append(context)
        
        return np.array(stacked_features)
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to zero mean and unit variance"""
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        return (features - mean) / std