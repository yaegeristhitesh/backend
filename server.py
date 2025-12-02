#!/usr/bin/env python3
"""
Standalone FastAPI server for Voice Phishing Detection API
"""
import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from typing import List
import uuid
import time
import logging
from datetime import datetime

# Import services
from src.services.audio_processors import AudioProcessor
from src.services.detector import PhishingDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Voice Phishing Detection API",
    version="1.0.0",
    description="AI-powered voice phishing detection system with biometric analysis",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
audio_processor = AudioProcessor()
detector = PhishingDetector()

@app.get("/")
async def root():
    return {
        "message": "Voice Phishing Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Voice Phishing Detection API",
        "timestamp": time.time(),
        "version": "1.0.0"
    }

@app.get("/api/v1/models/status")
async def get_model_status():
    """Get status of all detection models"""
    try:
        status_info = await detector.get_model_status()
        return {
            "status": "operational",
            "models": status_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/v1/analyze")
async def analyze_audio(
    audio_file: UploadFile = File(..., description="Audio file to analyze"),
    enable_biometric: bool = Form(True, description="Enable biometric check"),
    require_transcript: bool = Form(True, description="Include transcript in response")
):
    """
    Analyze audio file for phishing attempts.
    
    Supports multiple audio formats: WAV, MP3, M4A, FLAC, OGG
    Max file size: 50MB
    """
    start_time = datetime.now()
    request_id = str(uuid.uuid4())
    
    try:
        # Validate file size (50MB limit)
        max_size = 50 * 1024 * 1024  # 50MB
        content = await audio_file.read()
        if len(content) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: 50MB"
            )
        
        # Reset file pointer
        await audio_file.seek(0)
        
        logger.info(f"Processing audio: {audio_file.filename} [{request_id}]")
        
        # Process audio file
        audio_data = await audio_processor.process_upload(
            audio_file=audio_file,
            request_id=request_id
        )
        
        # Run phishing detection
        result = await detector.analyze(
            audio_data=audio_data,
            request_id=request_id
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        result["processing_time_ms"] = int(processing_time)
        
        # Log results
        logger.info(
            f"Analysis completed: {request_id} | "
            f"Phishing: {result['is_phishing']} | "
            f"Confidence: {result['overall_confidence']:.2f} | "
            f"Time: {processing_time:.0f}ms"
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed for {request_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )