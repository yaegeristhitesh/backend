from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Voice Phishing Detector"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # API
    API_V1_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # File uploads
    MAX_AUDIO_SIZE_MB: int = 50
    SUPPORTED_FORMATS: List[str] = [".wav", ".mp3", ".m4a", ".flac"]
    
    # Paths
    MODEL_DIR: str = "models"
    
    # ML Model settings (from your notebook)
    MAX_SEQUENCE_LENGTH: int = 100
    EMBEDDING_DIM: int = 300
    PREDICTION_THRESHOLD: float = 0.5
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()