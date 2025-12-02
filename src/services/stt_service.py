import whisper
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from dataclasses import dataclass

from src.core.config import settings

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
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_model()
        return cls._instance
    
    def _initialize_model(self):
        """Lazy load Whisper model"""
        if self._model is None:
            logger.info(f"Loading Whisper model: {settings.WHISPER_MODEL}")
            self._model = whisper.load_model(
                settings.WHISPER_MODEL,
                device=settings.WHISPER_DEVICE
            )
            logger.info(f"Whisper model loaded successfully")
    
    def transcribe(self, audio_array: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        """
        Transcribe audio to text.
        
        Args:
            audio_array: Audio signal as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            TranscriptionResult object
        """
        try:
            # Ensure audio is float32 and normalized
            audio = audio_array.astype(np.float32)
            if audio.max() > 1.0:
                audio = audio / np.max(np.abs(audio))
            
            # Transcribe with Whisper
            result = self._model.transcribe(
                audio,
                language=None,  # Auto-detect language
                task="transcribe",
                fp16=False,  # Use FP32 for CPU compatibility
                verbose=False
            )
            
            # Calculate average confidence from segments
            confidence = np.mean([seg.get('confidence', 0.5) for seg in result.get('segments', [])])
            
            return TranscriptionResult(
                text=result.get('text', ''),
                language=result.get('language', 'en'),
                confidence=float(confidence),
                segments=result.get('segments', []),
                word_timestamps=result.get('word_timestamps', None)
            )
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise Exception(f"Transcription error: {str(e)}")
    
    def detect_phishing_keywords(self, transcription: str) -> Dict[str, float]:
        """
        Scan transcription for common phishing keywords.
        """
        phishing_keywords = {
            # Urgency keywords
            'urgent': 0.8,
            'immediately': 0.9,
            'now': 0.7,
            'emergency': 0.9,
            'limited time': 0.85,
            
            # Financial keywords
            'bank account': 0.8,
            'password': 0.6,
            'credit card': 0.9,
            'social security': 0.95,
            'verify': 0.7,
            
            # Authority keywords
            'government': 0.6,
            'police': 0.7,
            'IRS': 0.9,
            'FBI': 0.8,
            
            # Threat keywords
            'arrest': 0.85,
            'lawsuit': 0.7,
            'fine': 0.75,
            'jail': 0.8,
        }
        
        text_lower = transcription.lower()
        detected = {}
        
        for keyword, weight in phishing_keywords.items():
            if keyword in text_lower:
                # Count occurrences
                count = text_lower.count(keyword)
                score = min(weight * (1 + 0.2 * (count - 1)), 1.0)
                detected[keyword] = score
        
        return detected