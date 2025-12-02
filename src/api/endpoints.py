from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional
import uuid
import asyncio
import logging
from datetime import datetime

from src.api.schemas import (
    AudioAnalysisRequest, 
    AudioAnalysisResponse,
    AudioFormat,
    BatchAnalysisResponse
)
from src.services.audio_processor import AudioProcessor
from src.services.detector import PhishingDetector
from src.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
audio_processor = AudioProcessor()
detector = PhishingDetector()

@router.post("/analyze", response_model=AudioAnalysisResponse)
async def analyze_audio(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(..., description="Audio file to analyze"),
    request_data: str = Form(default="{}", description="JSON configuration"),
):
    """
    Analyze audio file for phishing attempts.
    
    Supports multiple audio formats: WAV, MP3, M4A, FLAC
    Max file size: 50MB
    """
    start_time = datetime.now()
    request_id = str(uuid.uuid4())
    
    try:
        # Validate file size
        if audio_file.size > settings.MAX_AUDIO_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.MAX_AUDIO_SIZE_MB}MB"
            )
        
        # Parse request configuration
        import json
        try:
            config = json.loads(request_data)
            analysis_request = AudioAnalysisRequest(**config)
        except json.JSONDecodeError:
            analysis_request = AudioAnalysisRequest()
        
        logger.info(f"Processing audio: {audio_file.filename} [{request_id}]")
        
        # Process audio file
        audio_data = await audio_processor.process_upload(
            audio_file=audio_file,
            request_id=request_id
        )
        
        # Run phishing detection
        result = await detector.analyze(
            audio_data=audio_data,
            request_id=request_id,
            config=analysis_request
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        result.processing_time_ms = int(processing_time)
        
        # Log results
        logger.info(
            f"Analysis completed: {request_id} | "
            f"Phishing: {result.is_phishing} | "
            f"Confidence: {result.overall_confidence:.2f} | "
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

@router.post("/analyze/batch")
async def analyze_batch(
    audio_files: List[UploadFile] = File(..., description="Multiple audio files"),
    enable_biometric: bool = Form(True),
    priority: str = Form("normal"),
):
    """
    Analyze multiple audio files in batch mode.
    Maximum 10 files per batch.
    """
    if len(audio_files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 files allowed in batch mode"
        )
    
    batch_id = str(uuid.uuid4())
    logger.info(f"Starting batch analysis: {batch_id} with {len(audio_files)} files")
    
    results = []
    successful = 0
    failed = 0
    phishing_count = 0
    total_risk_score = 0.0
    
    # Process files concurrently
    async def process_single(file: UploadFile):
        try:
            request_id = str(uuid.uuid4())
            audio_data = await audio_processor.process_upload(file, request_id)
            
            # Create config for each file
            config = AudioAnalysisRequest(
                enable_biometric_check=enable_biometric,
                priority=priority
            )
            
            result = await detector.analyze(audio_data, request_id, config)
            return result
        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {str(e)}")
            return None
    
    # Run concurrent processing
    tasks = [process_single(file) for file in audio_files]
    analysis_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    for result in analysis_results:
        if isinstance(result, Exception) or result is None:
            failed += 1
            continue
            
        successful += 1
        results.append(result)
        
        if result.is_phishing:
            phishing_count += 1
        
        total_risk_score += result.risk_score
    
    # Calculate averages
    avg_risk = total_risk_score / successful if successful > 0 else 0
    
    return BatchAnalysisResponse(
        batch_id=batch_id,
        total_processed=len(audio_files),
        successful_analyses=successful,
        failed_analyses=failed,
        phishing_count=phishing_count,
        average_risk_score=avg_risk,
        results=results
    )

@router.post("/scammer/enroll")
async def enroll_scammer_voice(
    speaker_id: str = Form(..., description="Unique identifier for scammer"),
    audio_file: UploadFile = File(..., description="Voice sample"),
    metadata: str = Form(default="{}", description="Additional metadata"),
):
    """
    Enroll a known scammer's voice into the biometric database.
    """
    try:
        # Process audio
        request_id = str(uuid.uuid4())
        audio_data = await audio_processor.process_upload(audio_file, request_id)
        
        # Parse metadata
        import json
        metadata_dict = json.loads(metadata) if metadata else {}
        
        # Enroll in biometric system
        from src.services.voice_biometric import VoiceBiometricService
        result = await VoiceBiometricService.enroll_scammer(
            speaker_id=speaker_id,
            audio_features=audio_data["features"],
            metadata=metadata_dict
        )
        
        return {
            "status": "success",
            "speaker_id": speaker_id,
            "biometric_id": result.get("biometric_id"),
            "confidence": result.get("confidence"),
            "message": "Scammer voice enrolled successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to enroll scammer: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/status")
async def get_model_status():
    """
    Get status of all detection models.
    """
    status_info = await detector.get_model_status()
    return {
        "status": "operational",
        "models": status_info,
        "total_models": len(status_info),
        "enabled_models": settings.ENABLED_MODELS,
        "timestamp": datetime.now().isoformat()
    }