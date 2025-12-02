from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class AudioFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"
    M4A = "m4a"
    FLAC = "flac"

class AnalysisPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    REALTIME = "realtime"

class WarningSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class BiometricMatch(BaseModel):
    speaker_id: Optional[str] = None
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_known_scammer: bool = False
    scammer_database_match: Optional[str] = None

class PhishingWarning(BaseModel):
    model: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    severity: WarningSeverity
    indicators: List[str]
    explanation: str
    recommendation: Optional[str] = None
    
    @validator('severity')
    def validate_severity(cls, v, values):
        confidence = values.get('confidence', 0)
        if confidence > 0.8 and v == WarningSeverity.LOW:
            return WarningSeverity.MEDIUM
        return v

class AudioAnalysisRequest(BaseModel):
    audio_format: AudioFormat = AudioFormat.MP3
    priority: AnalysisPriority = AnalysisPriority.NORMAL
    enable_biometric_check: bool = True
    require_transcript: bool = True
    callback_url: Optional[str] = None
    
    @validator('callback_url')
    def validate_callback_url(cls, v):
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError('Callback URL must start with http:// or https://')
        return v

class AudioAnalysisResponse(BaseModel):
    request_id: str
    status: str
    transcript: Optional[str] = None
    is_phishing: bool
    overall_confidence: float
    processing_time_ms: int
    warnings: List[PhishingWarning]
    biometric_match: Optional[BiometricMatch] = None
    model_breakdown: Dict[str, float]
    audio_metadata: Dict[str, Any]
    risk_score: float = Field(..., ge=0.0, le=1.0)
    
    @property
    def is_critical(self) -> bool:
        return self.risk_score >= 0.8

class BatchAnalysisResponse(BaseModel):
    batch_id: str
    total_processed: int
    successful_analyses: int
    failed_analyses: int
    phishing_count: int
    average_risk_score: float
    results: List[AudioAnalysisResponse]