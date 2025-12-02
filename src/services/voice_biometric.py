import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import pickle
import os
import logging
from dataclasses import dataclass
from datetime import datetime
from scipy.spatial.distance import cosine, euclidean
import hashlib

from src.core.config import settings
from src.models.cnn_speaker import VoiceCNN

logger = logging.getLogger(__name__)

@dataclass
class VoicePrint:
    """Voice biometric template"""
    speaker_id: str
    embedding: np.ndarray
    metadata: Dict
    confidence: float
    created_at: datetime
    scammer_flag: bool = False
    scam_count: int = 0
    
    def get_hash(self) -> str:
        """Get unique hash for this voiceprint"""
        data = f"{self.speaker_id}_{self.embedding.tobytes().hex()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

class VoiceBiometricService:
    """CNN-based voice biometric system (implementing paper concepts)"""
    
    _instance = None
    _model = None
    _database = {}  # speaker_id -> VoicePrint
    _scammer_db = {}  # biometric_hash -> scammer_data
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def initialize(cls):
        """Initialize the service"""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load_model()
            cls._instance._load_database()
    
    def _load_model(self):
        """Load CNN model for voice recognition"""
        self.model = VoiceCNN(
            num_speakers=50,
            feature_dim=settings.MFCC_FEATURES * 4  # MFCC + delta + delta-delta + filterbank
        )
        
        # Load pre-trained weights if available
        model_path = "models/voice_cnn.pth"
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            logger.info("Loaded pre-trained CNN model")
        
        self.model.eval()
        
    def _load_database(self):
        """Load voiceprint database from file"""
        db_path = "data/voiceprints.pkl"
        if os.path.exists(db_path):
            try:
                with open(db_path, 'rb') as f:
                    self._database = pickle.load(f)
                logger.info(f"Loaded {len(self._database)} voiceprints from database")
            except Exception as e:
                logger.error(f"Failed to load database: {e}")
    
    def _save_database(self):
        """Save voiceprint database to file"""
        os.makedirs("data", exist_ok=True)
        db_path = "data/voiceprints.pkl"
        try:
            with open(db_path, 'wb') as f:
                pickle.dump(self._database, f)
            logger.info(f"Saved {len(self._database)} voiceprints to database")
        except Exception as e:
            logger.error(f"Failed to save database: {e}")
    
    def extract_embedding(self, mfcc_features: np.ndarray) -> np.ndarray:
        """
        Extract voice embedding using CNN model.
        
        Args:
            mfcc_features: MFCC features (n_frames, n_features)
            
        Returns:
            Normalized embedding vector
        """
        try:
            # Ensure correct shape
            if len(mfcc_features.shape) == 2:
                # Add batch dimension and channel dimension
                features_tensor = torch.FloatTensor(mfcc_features).unsqueeze(0).unsqueeze(0)
            else:
                features_tensor = torch.FloatTensor(mfcc_features).unsqueeze(0)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model.extract_embedding(features_tensor)
                embedding = embedding.numpy().flatten()
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}")
            raise
    
    async def enroll_speaker(self, speaker_id: str, audio_features: Dict, 
                           metadata: Optional[Dict] = None) -> VoicePrint:
        """
        Enroll a new speaker in the system.
        """
        # Extract MFCC features
        mfcc = audio_features.get("mfcc")
        if mfcc is None or len(mfcc) == 0:
            raise ValueError("No MFCC features available")
        
        # Extract embedding
        embedding = self.extract_embedding(mfcc)
        
        # Calculate enrollment quality
        confidence = self._calculate_enrollment_confidence(embedding)
        
        # Create voiceprint
        voiceprint = VoicePrint(
            speaker_id=speaker_id,
            embedding=embedding,
            metadata=metadata or {},
            confidence=confidence,
            created_at=datetime.now(),
            scammer_flag=False
        )
        
        # Store in database
        self._database[speaker_id] = voiceprint
        self._save_database()
        
        logger.info(f"Enrolled speaker: {speaker_id} (confidence: {confidence:.3f})")
        return voiceprint
    
    async def enroll_scammer(self, speaker_id: str, audio_features: Dict,
                           metadata: Optional[Dict] = None) -> VoicePrint:
        """
        Enroll a known scammer into the database.
        """
        voiceprint = await self.enroll_speaker(speaker_id, audio_features, metadata)
        voiceprint.scammer_flag = True
        
        # Add to scammer-specific database
        voice_hash = voiceprint.get_hash()
        self._scammer_db[voice_hash] = {
            "speaker_id": speaker_id,
            "embedding": voiceprint.embedding,
            "metadata": voiceprint.metadata,
            "enrollment_date": datetime.now()
        }
        
        logger.warning(f"Enrolled scammer: {speaker_id}")
        return voiceprint
    
    async def verify_speaker(self, claimed_id: str, audio_features: Dict) -> Dict:
        """
        Verify if audio matches claimed speaker.
        """
        if claimed_id not in self._database:
            return {
                "verified": False,
                "confidence": 0.0,
                "error": "Speaker not enrolled",
                "similarity": 0.0
            }
        
        # Get stored voiceprint
        stored = self._database[claimed_id]
        
        # Extract embedding from current audio
        mfcc = audio_features.get("mfcc")
        if mfcc is None:
            return {
                "verified": False,
                "confidence": 0.0,
                "error": "No audio features",
                "similarity": 0.0
            }
        
        current_embedding = self.extract_embedding(mfcc)
        
        # Calculate similarity (cosine similarity)
        similarity = 1 - cosine(current_embedding, stored.embedding)
        
        # Calculate verification confidence
        confidence = self._calculate_verification_confidence(
            similarity, 
            stored.confidence
        )
        
        # Check against threshold
        verified = similarity >= settings.CONFIDENCE_THRESHOLD
        
        return {
            "verified": verified,
            "similarity": float(similarity),
            "confidence": float(confidence),
            "threshold": settings.CONFIDENCE_THRESHOLD,
            "speaker_id": claimed_id
        }
    
    async def identify_speaker(self, audio_features: Dict) -> Dict:
        """
        Identify speaker from audio (1:N matching).
        """
        mfcc = audio_features.get("mfcc")
        if mfcc is None or len(self._database) == 0:
            return {
                "identified": False,
                "matches": [],
                "best_match": None
            }
        
        current_embedding = self.extract_embedding(mfcc)
        
        # Compare against all enrolled speakers
        matches = []
        for speaker_id, voiceprint in self._database.items():
            similarity = 1 - cosine(current_embedding, voiceprint.embedding)
            
            if similarity >= settings.CONFIDENCE_THRESHOLD:
                matches.append({
                    "speaker_id": speaker_id,
                    "similarity": float(similarity),
                    "is_scammer": voiceprint.scammer_flag,
                    "enrollment_date": voiceprint.created_at.isoformat()
                })
        
        # Sort by similarity
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        
        return {
            "identified": len(matches) > 0,
            "matches": matches,
            "best_match": matches[0] if matches else None,
            "total_comparisons": len(self._database)
        }
    
    async def detect_known_scammer(self, audio_features: Dict) -> Dict:
        """
        Check if voice matches any known scammer.
        """
        mfcc = audio_features.get("mfcc")
        if mfcc is None or len(self._scammer_db) == 0:
            return {
                "is_scammer": False,
                "confidence": 0.0,
                "matches": []
            }
        
        current_embedding = self.extract_embedding(mfcc)
        
        scammer_matches = []
        for scammer_hash, scammer_data in self._scammer_db.items():
            scammer_embedding = scammer_data["embedding"]
            similarity = 1 - cosine(current_embedding, scammer_embedding)
            
            if similarity >= settings.SCAM_THRESHOLD:
                scammer_matches.append({
                    "scammer_id": scammer_data["speaker_id"],
                    "similarity": float(similarity),
                    "metadata": scammer_data.get("metadata", {}),
                    "hash": scammer_hash
                })
        
        is_scammer = len(scammer_matches) > 0
        
        return {
            "is_scammer": is_scammer,
            "confidence": max([m["similarity"] for m in scammer_matches]) if scammer_matches else 0.0,
            "matches": scammer_matches,
            "total_scammers_checked": len(self._scammer_db)
        }
    
    def _calculate_enrollment_confidence(self, embedding: np.ndarray) -> float:
        """Calculate confidence in enrollment quality"""
        # Check embedding norm
        norm = np.linalg.norm(embedding)
        if norm < 0.1:
            return 0.3
        
        # Check embedding stability (if we had multiple samples)
        return min(0.7 + norm * 0.3, 1.0)
    
    def _calculate_verification_confidence(self, similarity: float, 
                                         enrollment_confidence: float) -> float:
        """Calculate overall verification confidence"""
        # Weight similarity more heavily
        return 0.7 * similarity + 0.3 * enrollment_confidence
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        total = len(self._database)
        scammers = sum(1 for v in self._database.values() if v.scammer_flag)
        
        return {
            "total_speakers": total,
            "known_scammers": scammers,
            "legitimate_speakers": total - scammers,
            "average_confidence": np.mean([v.confidence for v in self._database.values()]) if total > 0 else 0
        }