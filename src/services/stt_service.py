import whisper
import numpy as np
from typing import Dict, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TranscriptionResult:
    text: str
    language: str
    confidence: float
    segments: list
    word_timestamps: Optional[list] = None

class STTService:
    """Speech-to-Text service using OpenAI Whisper"""
    
    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model"""
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe_file(self, audio_path: str) -> TranscriptionResult:
        """Transcribe audio file to text"""
        try:
            result = self.model.transcribe(audio_path, fp16=False)
            
            # Calculate average confidence from segments
            segments = result.get('segments', [])
            if segments:
                confidence = np.mean([seg.get('no_speech_prob', 0.5) for seg in segments])
                confidence = 1.0 - confidence  # Convert no_speech_prob to confidence
            else:
                confidence = 0.5
            
            return TranscriptionResult(
                text=result.get('text', '').strip(),
                language=result.get('language', 'en'),
                confidence=float(confidence),
                segments=segments
            )
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise Exception(f"Transcription error: {e}")
    
    def transcribe_array(self, audio_array: np.ndarray) -> TranscriptionResult:
        """Transcribe audio array to text"""
        try:
            # Ensure audio is float32 and normalized
            audio = audio_array.astype(np.float32)
            if audio.max() > 1.0:
                audio = audio / np.max(np.abs(audio))
            
            result = self.model.transcribe(audio, fp16=False)
            
            segments = result.get('segments', [])
            if segments:
                confidence = np.mean([1.0 - seg.get('no_speech_prob', 0.5) for seg in segments])
            else:
                confidence = 0.5
            
            return TranscriptionResult(
                text=result.get('text', '').strip(),
                language=result.get('language', 'en'),
                confidence=float(confidence),
                segments=segments
            )
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise Exception(f"Transcription error: {e}")
    
    def detect_phishing_keywords(self, transcription: str) -> Dict[str, float]:
        """Scan transcription for common phishing keywords"""
        phishing_keywords = {
            # Urgency keywords
            'urgent': 0.8, 'immediately': 0.9, 'now': 0.7, 'emergency': 0.9,
            'limited time': 0.85, 'expires': 0.7, 'deadline': 0.8,
            
            # Financial keywords
            'bank account': 0.8, 'password': 0.6, 'credit card': 0.9,
            'social security': 0.95, 'verify': 0.7, 'confirm': 0.6,
            'update': 0.5, 'suspend': 0.8, 'freeze': 0.8,
            
            # Authority keywords
            'government': 0.6, 'police': 0.7, 'irs': 0.9, 'fbi': 0.8,
            'court': 0.7, 'legal': 0.6, 'official': 0.5,
            
            # Threat keywords
            'arrest': 0.85, 'lawsuit': 0.7, 'fine': 0.75, 'jail': 0.8,
            'penalty': 0.7, 'charges': 0.8, 'investigation': 0.7,
            
            # Action keywords
            'click': 0.6, 'call back': 0.7, 'press': 0.5, 'transfer': 0.8,
            'provide': 0.6, 'give me': 0.7, 'tell me': 0.6
        }
        
        text_lower = transcription.lower()
        detected = {}
        
        for keyword, weight in phishing_keywords.items():
            if keyword in text_lower:
                count = text_lower.count(keyword)
                score = min(weight * (1 + 0.2 * (count - 1)), 1.0)
                detected[keyword] = score
        
        return detected