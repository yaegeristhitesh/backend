import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

from ..model.cnn_speaker import SpeakerEmbeddingExtractor

logger = logging.getLogger(__name__)

class VoiceBiometricService:
    """Voice biometric service for scammer identification"""
    
    def __init__(self, database_path: str = "data/scammer_db.json"):
        self.database_path = Path(database_path)
        self.embedding_extractor = SpeakerEmbeddingExtractor()
        self.scammer_database = self._load_database()
        self.similarity_threshold = 0.75  # Threshold for scammer match
        
    def _load_database(self) -> Dict:
        """Load scammer voice database"""
        try:
            if self.database_path.exists():
                with open(self.database_path, 'r') as f:
                    db = json.load(f)
                logger.info(f"Loaded scammer database with {len(db)} entries")
                return db
            else:
                logger.info("Creating new scammer database")
                # Create directory if it doesn't exist
                self.database_path.parent.mkdir(parents=True, exist_ok=True)
                return {}
        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            return {}
    
    def _save_database(self):
        """Save scammer database to file"""
        try:
            with open(self.database_path, 'w') as f:
                json.dump(self.scammer_database, f, indent=2)
            logger.info("Scammer database saved")
        except Exception as e:
            logger.error(f"Failed to save database: {e}")
    
    def analyze_voice(self, mfcc_features: List[List[float]], request_id: str = None) -> Dict:
        """
        Analyze voice for known scammer patterns
        
        Args:
            mfcc_features: MFCC features from audio
            request_id: Request ID for tracking
            
        Returns:
            Analysis results with scammer match info
        """
        try:
            # Convert to numpy array
            mfcc_array = np.array(mfcc_features)
            
            if mfcc_array.size == 0:
                return {
                    "is_known_scammer": False,
                    "confidence": 0.0,
                    "matched_scammer_id": None,
                    "similarity_score": 0.0,
                    "error": "Empty MFCC features",
                    "embedding": None
                }
            
            # Extract speaker embedding
            embedding = self.embedding_extractor.extract_embedding(mfcc_array)
            
            # Check against known scammers
            best_match = self._find_best_match(embedding)
            
            if best_match:
                scammer_id, similarity = best_match
                is_scammer = similarity >= self.similarity_threshold
                
                # Update activity if matched
                if is_scammer:
                    self.update_scammer_activity(scammer_id)
                
                return {
                    "is_known_scammer": is_scammer,
                    "confidence": similarity,
                    "matched_scammer_id": scammer_id if is_scammer else None,
                    "similarity_score": similarity,
                    "threshold": self.similarity_threshold,
                    "embedding_dim": len(embedding),
                    "embedding": embedding.tolist()  # For adaptive learning
                }
            else:
                return {
                    "is_known_scammer": False,
                    "confidence": 0.0,
                    "matched_scammer_id": None,
                    "similarity_score": 0.0,
                    "threshold": self.similarity_threshold,
                    "embedding_dim": len(embedding),
                    "embedding": embedding.tolist()  # For adaptive learning
                }
                
        except Exception as e:
            logger.error(f"Voice analysis failed: {e}")
            return {
                "is_known_scammer": False,
                "confidence": 0.0,
                "matched_scammer_id": None,
                "similarity_score": 0.0,
                "error": str(e),
                "embedding": None
            }
    
    def _find_best_match(self, embedding: np.ndarray) -> Optional[Tuple[str, float]]:
        """Find best matching scammer in database"""
        if not self.scammer_database:
            return None
        
        best_similarity = 0.0
        best_scammer_id = None
        
        for scammer_id, scammer_data in self.scammer_database.items():
            stored_embedding = np.array(scammer_data["embedding"])
            similarity = self.embedding_extractor.compute_similarity(embedding, stored_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_scammer_id = scammer_id
        
        return (best_scammer_id, best_similarity) if best_similarity > 0 else None
    
    def add_scammer(self, scammer_id: str, mfcc_features: List[List[float]], 
                   metadata: Dict = None) -> bool:
        """
        Add a new scammer to the database
        
        Args:
            scammer_id: Unique identifier for the scammer
            mfcc_features: MFCC features from scammer's voice
            metadata: Additional information about the scammer
            
        Returns:
            Success status
        """
        try:
            mfcc_array = np.array(mfcc_features)
            embedding = self.embedding_extractor.extract_embedding(mfcc_array)
            
            self.scammer_database[scammer_id] = {
                "embedding": embedding.tolist(),
                "metadata": metadata or {},
                "added_timestamp": str(np.datetime64('now')),
                "call_count": 1
            }
            
            self._save_database()
            logger.info(f"Added scammer {scammer_id} to database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add scammer: {e}")
            return False
    
    def update_scammer_activity(self, scammer_id: str):
        """Update activity count for known scammer"""
        if scammer_id in self.scammer_database:
            self.scammer_database[scammer_id]["call_count"] += 1
            self.scammer_database[scammer_id]["last_seen"] = str(np.datetime64('now'))
            self._save_database()
    
    def get_scammer_stats(self) -> Dict:
        """Get statistics about known scammers"""
        if not self.scammer_database:
            return {
                "total_scammers": 0,
                "total_calls": 0,
                "most_active": None
            }
        
        total_calls = sum(data.get("call_count", 1) for data in self.scammer_database.values())
        most_active = max(
            self.scammer_database.items(),
            key=lambda x: x[1].get("call_count", 1)
        )
        
        return {
            "total_scammers": len(self.scammer_database),
            "total_calls": total_calls,
            "most_active": {
                "id": most_active[0],
                "calls": most_active[1].get("call_count", 1)
            }
        }
    
    def adaptive_learn_scammer(self, embedding: List[float], scam_type: str = "Unknown", 
                              confidence_threshold: float = 0.8) -> bool:
        """
        Adaptively learn new scammer from high-confidence phishing detection
        
        Args:
            embedding: Voice embedding from confirmed phishing call
            scam_type: Type of scam detected
            confidence_threshold: Minimum confidence to add to database
            
        Returns:
            True if scammer was added to database
        """
        try:
            embedding_array = np.array(embedding)
            
            # Check if this voice is already similar to known scammers
            best_match = self._find_best_match(embedding_array)
            
            if best_match and best_match[1] >= self.similarity_threshold:
                # Already known scammer, just update activity
                self.update_scammer_activity(best_match[0])
                logger.info(f"Updated activity for known scammer: {best_match[0]}")
                return False
            
            # Generate new scammer ID
            scammer_count = len([k for k in self.scammer_database.keys() if k.startswith('adaptive_')])
            new_scammer_id = f"adaptive_scammer_{scammer_count + 1:03d}"
            
            # Add to database
            self.scammer_database[new_scammer_id] = {
                "embedding": embedding,
                "metadata": {
                    "description": f"Adaptively learned scammer - {scam_type}",
                    "scam_type": scam_type,
                    "learning_method": "adaptive",
                    "first_detected": str(np.datetime64('now'))
                },
                "added_timestamp": str(np.datetime64('now')),
                "call_count": 1,
                "last_seen": str(np.datetime64('now'))
            }
            
            self._save_database()
            logger.info(f"Adaptively learned new scammer: {new_scammer_id} ({scam_type})")
            return True
            
        except Exception as e:
            logger.error(f"Adaptive learning failed: {e}")
            return False
    
    def get_service_info(self) -> Dict:
        """Get service information"""
        adaptive_count = len([k for k in self.scammer_database.keys() if k.startswith('adaptive_')])
        
        return {
            "service_name": "Voice Biometric Service (Adaptive)",
            "model_info": self.embedding_extractor.get_model_info(),
            "database_stats": self.get_scammer_stats(),
            "similarity_threshold": self.similarity_threshold,
            "database_path": str(self.database_path),
            "adaptive_learning": {
                "enabled": True,
                "learned_scammers": adaptive_count,
                "total_scammers": len(self.scammer_database)
            }
        }