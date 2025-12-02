import asyncio
from typing import Dict, List, Optional
import logging
from datetime import datetime

from src.api.schemas import AudioAnalysisRequest, AudioAnalysisResponse
from src.services.stt_service import STTService
from src.models.phishing_models import (
    LinguisticAnalyzer, 
    URLDetector, 
    UrgencyScorer
)
from src.services.voice_biometric import VoiceBiometricService
from src.core.config import settings

logger = logging.getLogger(__name__)

class PhishingDetector:
    """Main phishing detection orchestrator"""
    
    def __init__(self):
        self.stt_service = STTService()
        self.linguistic_analyzer = LinguisticAnalyzer()
        self.url_detector = URLDetector()
        self.urgency_scorer = UrgencyScorer()
        self.biometric_service = VoiceBiometricService()
        
        self.models = {
            "linguistic": self.linguistic_analyzer,
            "url_detector": self.url_detector,
            "urgency_scorer": self.urgency_scorer,
        }
    
    async def analyze(self, audio_data: Dict, request_id: str, 
                     config: AudioAnalysisRequest) -> AudioAnalysisResponse:
        """
        Main analysis pipeline.
        """
        logger.info(f"[{request_id}] Starting analysis pipeline")
        
        # Extract audio array for STT
        audio_array = audio_data.get("audio_array")
        if audio_array is None:
            raise ValueError("No audio data available")
        
        # Step 1: Speech-to-Text
        logger.debug(f"[{request_id}] Starting transcription")
        transcription_result = self.stt_service.transcribe(
            audio_array,
            sample_rate=audio_data["sample_rate"]
        )
        
        transcript = transcription_result.text if config.require_transcript else None
        
        # Step 2: Run detection models in parallel
        logger.debug(f"[{request_id}] Running detection models")
        model_tasks = []
        
        # Add enabled models to tasks
        for model_name, model in self.models.items():
            if model_name in settings.ENABLED_MODELS:
                task = asyncio.create_task(
                    self._run_model(model, model_name, transcription_result.text)
                )
                model_tasks.append((model_name, task))
        
        # Step 3: Voice biometric check (if enabled)
        biometric_result = None
        if config.enable_biometric_check:
            logger.debug(f"[{request_id}] Running biometric check")
            biometric_task = asyncio.create_task(
                self.biometric_service.detect_known_scammer(audio_data["features"])
            )
        else:
            biometric_task = None
        
        # Wait for all models to complete
        model_results = {}
        for model_name, task in model_tasks:
            try:
                result = await task
                model_results[model_name] = result
            except Exception as e:
                logger.error(f"[{request_id}] Model {model_name} failed: {e}")
                model_results[model_name] = self._create_failed_result(model_name)
        
        # Wait for biometric result if enabled
        if biometric_task:
            try:
                biometric_result = await biometric_task
            except Exception as e:
                logger.error(f"[{request_id}] Biometric check failed: {e}")
                biometric_result = {"is_scammer": False, "confidence": 0.0}
        
        # Step 4: Aggregate results
        logger.debug(f"[{request_id}] Aggregating results")
        aggregated = self._aggregate_results(
            model_results, 
            biometric_result,
            transcription_result.confidence
        )
        
        # Step 5: Create response
        response = AudioAnalysisResponse(
            request_id=request_id,
            status="completed",
            transcript=transcript,
            is_phishing=aggregated["is_phishing"],
            overall_confidence=aggregated["overall_confidence"],
            processing_time_ms=0,  # Will be set by endpoint
            warnings=aggregated["warnings"],
            biometric_match=aggregated.get("biometric_match"),
            model_breakdown=aggregated["model_scores"],
            audio_metadata={
                "duration": audio_data["duration_seconds"],
                "sample_rate": audio_data["sample_rate"],
                "format": audio_data["original_format"],
                "speech_ratio": audio_data["features"].get("speech_ratio", 0)
            },
            risk_score=aggregated["risk_score"]
        )
        
        logger.info(f"[{request_id}] Analysis complete - Phishing: {response.is_phishing}")
        return response
    
    async def _run_model(self, model, model_name: str, text: str):
        """Run a single detection model"""
        try:
            result = model.analyze(text)
            return result
        except Exception as e:
            logger.error(f"Model {model_name} failed: {e}")
            return self._create_failed_result(model_name)
    
    def _create_failed_result(self, model_name: str):
        """Create a fallback result for failed models"""
        from src.models.phishing_models import DetectionResult
        
        return DetectionResult(
            model_name=model_name,
            confidence=0.0,
            indicators=["Model execution failed"],
            explanation=f"{model_name} model failed to execute",
            metadata={"status": "failed"}
        )
    
    def _aggregate_results(self, model_results: Dict, biometric_result: Optional[Dict], 
                          transcription_confidence: float) -> Dict:
        """
        Aggregate results from all models into final decision.
        """
        warnings = []
        model_scores = {}
        total_confidence = 0.0
        active_models = 0
        
        # Process each model's results
        for model_name, result in model_results.items():
            if result.confidence > 0:
                # Convert DetectionResult to PhishingWarning
                from src.api.schemas import PhishingWarning, WarningSeverity
                
                # Determine severity based on confidence
                if result.confidence >= 0.8:
                    severity = WarningSeverity.HIGH
                elif result.confidence >= 0.6:
                    severity = WarningSeverity.MEDIUM
                else:
                    severity = WarningSeverity.LOW
                
                warning = PhishingWarning(
                    model=model_name,
                    confidence=result.confidence,
                    severity=severity,
                    indicators=result.indicators[:5],  # Limit indicators
                    explanation=result.explanation,
                    recommendation=self._get_recommendation(model_name, result.confidence)
                )
                warnings.append(warning)
                
                # Add to scores
                model_scores[model_name] = result.confidence
                total_confidence += result.confidence
                active_models += 1
        
        # Add biometric warning if scammer detected
        biometric_match = None
        if biometric_result and biometric_result.get("is_scammer"):
            from src.api.schemas import BiometricMatch, PhishingWarning, WarningSeverity
            
            scammer_matches = biometric_result.get("matches", [])
            best_match = scammer_matches[0] if scammer_matches else None
            
            if best_match:
                # Create biometric warning
                warnings.append(PhishingWarning(
                    model="voice_biometric",
                    confidence=best_match["similarity"],
                    severity=WarningSeverity.CRITICAL,
                    indicators=[f"Known scammer: {best_match['scammer_id']}"],
                    explanation="Voice matches known scammer in database",
                    recommendation="Terminate call immediately and report"
                ))
                
                # Add to scores with high weight
                model_scores["voice_biometric"] = best_match["similarity"]
                total_confidence += best_match["similarity"] * 2  # Double weight for scammers
                active_models += 2  # Count biometric as double
                
                # Create biometric match info
                biometric_match = BiometricMatch(
                    speaker_id=best_match["scammer_id"],
                    similarity_score=best_match["similarity"],
                    confidence=best_match["similarity"],
                    is_known_scammer=True,
                    scammer_database_match=best_match["scammer_id"]
                )
        
        # Calculate overall metrics
        overall_confidence = total_confidence / active_models if active_models > 0 else 0
        
        # Adjust confidence based on transcription quality
        if transcription_confidence < 0.5:
            overall_confidence *= 0.8  # Penalize for poor transcription
        
        # Calculate risk score (weighted combination)
        risk_score = self._calculate_risk_score(
            model_scores, 
            biometric_result, 
            overall_confidence
        )
        
        # Determine if phishing
        is_phishing = (
            risk_score >= settings.SCAM_THRESHOLD or 
            (biometric_result and biometric_result.get("is_scammer"))
        )
        
        return {
            "is_phishing": is_phishing,
            "overall_confidence": overall_confidence,
            "risk_score": risk_score,
            "warnings": warnings,
            "model_scores": model_scores,
            "biometric_match": biometric_match
        }
    
    def _get_recommendation(self, model_name: str, confidence: float) -> str:
        """Get recommendation based on model and confidence"""
        recommendations = {
            "linguistic_analyzer": {
                "high": "Be cautious of pressure tactics and verify information",
                "medium": "Verify the source before taking any action",
                "low": "Standard communication, stay vigilant"
            },
            "url_detector": {
                "high": "Do not click any links, verify website independently",
                "medium": "Check URL legitimacy before clicking",
                "low": "Links appear legitimate but verify if unsure"
            },
            "urgency_scorer": {
                "high": "High-pressure tactics detected, likely scam",
                "medium": "Moderate urgency, verify before acting",
                "low": "Normal communication pace"
            }
        }
        
        if confidence >= 0.8:
            level = "high"
        elif confidence >= 0.5:
            level = "medium"
        else:
            level = "low"
        
        return recommendations.get(model_name, {}).get(level, "Use caution and verify information")
    
    def _calculate_risk_score(self, model_scores: Dict, biometric_result: Optional[Dict], 
                            base_confidence: float) -> float:
        """Calculate overall risk score"""
        weights = {
            "linguistic_analyzer": 0.3,
            "url_detector": 0.4,
            "urgency_scorer": 0.3,
            "voice_biometric": 0.8  # High weight for biometric matches
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for model, score in model_scores.items():
            weight = weights.get(model, 0.2)
            weighted_sum += score * weight
            total_weight += weight
        
        # Add biometric penalty if scammer detected
        if biometric_result and biometric_result.get("is_scammer"):
            scam_confidence = biometric_result.get("confidence", 0)
            weighted_sum += scam_confidence * weights["voice_biometric"]
            total_weight += weights["voice_biometric"]
        
        risk_score = weighted_sum / total_weight if total_weight > 0 else base_confidence
        
        # Apply non-linear scaling for high scores
        if risk_score > 0.7:
            risk_score = 0.7 + (risk_score - 0.7) * 1.5
        
        return min(risk_score, 1.0)
    
    async def get_model_status(self) -> Dict:
        """Get status of all detection models"""
        status = {}
        
        for model_name in settings.ENABLED_MODELS:
            status[model_name] = {
                "enabled": True,
                "status": "operational",
                "description": self._get_model_description(model_name)
            }
        
        # Add biometric service status
        status["voice_biometric"] = {
            "enabled": True,
            "status": "operational",
            "description": "CNN-based voice biometric verification",
            "database_size": len(VoiceBiometricService()._database) if hasattr(VoiceBiometricService(), '_database') else 0
        }
        
        return status
    
    def _get_model_description(self, model_name: str) -> str:
        """Get description for a model"""
        descriptions = {
            "linguistic": "Analyzes text for phishing linguistic patterns and keywords",
            "url_detector": "Detects suspicious URLs and domain spoofing attempts",
            "urgency_scorer": "Identifies urgency and pressure tactics common in scams",
            "voice_biometric": "CNN-based voice recognition for known scammer detection"
        }
        return descriptions.get(model_name, "Unknown model")