import asyncio
from typing import Dict, List, Optional
import logging
from datetime import datetime

from .stt_service import STTService
from .ml_model_service import MLModelService
from .voice_biometric import VoiceBiometricService

logger = logging.getLogger(__name__)

class PhishingDetector:
    """Main phishing detection orchestrator"""
    
    def __init__(self):
        self.stt_service = STTService()
        self.ml_model_service = MLModelService()
        self.biometric_service = VoiceBiometricService()
        
    async def analyze(self, audio_data: Dict, request_id: str) -> Dict:
        """Main analysis pipeline"""
        logger.info(f"[{request_id}] Starting analysis pipeline")
        
        try:
            # Step 1: Speech-to-Text
            logger.debug(f"[{request_id}] Starting transcription")
            audio_array = audio_data.get("audio_array")
            if audio_array is None:
                raise ValueError("No audio data available")
            
            transcription_result = self.stt_service.transcribe_array(audio_array)
            transcript = transcription_result.text
            
            # Step 2: Run detection models in parallel
            logger.debug(f"[{request_id}] Running detection models")
            
            # ML Model prediction
            ml_task = asyncio.create_task(
                self._run_ml_model(transcript)
            )
            
            # Rule-based detection
            rule_task = asyncio.create_task(
                self._run_rule_based_detection(transcript)
            )
            
            # Voice biometric check
            biometric_task = asyncio.create_task(
                self._run_biometric_check(audio_data["features"], request_id)
            )
            
            # Wait for all tasks
            ml_result, rule_result, biometric_result = await asyncio.gather(
                ml_task, rule_task, biometric_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(ml_result, Exception):
                logger.error(f"ML model failed: {ml_result}")
                ml_result = {"confidence": 0.0, "is_phishing": False, "error": str(ml_result)}
            
            if isinstance(rule_result, Exception):
                logger.error(f"Rule-based detection failed: {rule_result}")
                rule_result = {"confidence": 0.0, "warnings": [], "error": str(rule_result)}
            
            if isinstance(biometric_result, Exception):
                logger.error(f"Biometric check failed: {biometric_result}")
                biometric_result = {"is_known_scammer": False, "confidence": 0.0, "error": str(biometric_result)}
            
            # Step 3: Aggregate results
            logger.debug(f"[{request_id}] Aggregating results")
            aggregated = self._aggregate_results(
                ml_result, rule_result, biometric_result, transcription_result.confidence
            )
            
            # Adaptive learning: If high-confidence phishing detected, learn the voice pattern
            if (aggregated["is_phishing"] and aggregated["overall_confidence"] >= 0.7 and 
                biometric_result.get('embedding') and 
                not biometric_result.get('is_known_scammer', False)):
                
                # Determine scam type from rule-based detection
                scam_type = "Unknown"
                if rule_result.get('keywords_detected'):
                    keywords = rule_result['keywords_detected']
                    if any('irs' in k.lower() for k in keywords):
                        scam_type = "IRS Scam"
                    elif any('bank' in k.lower() for k in keywords):
                        scam_type = "Banking Fraud"
                    elif any(k.lower() in ['microsoft', 'tech', 'computer'] for k in keywords):
                        scam_type = "Tech Support Scam"
                    else:
                        scam_type = "General Phishing"
                
                # Learn this scammer's voice
                learned = self.biometric_service.adaptive_learn_scammer(
                    biometric_result['embedding'], scam_type, aggregated["overall_confidence"]
                )
                
                if learned:
                    logger.info(f"[{request_id}] Adaptively learned new scammer: {scam_type}")
            
            # Step 4: Create response
            response = {
                "request_id": request_id,
                "status": "completed",
                "transcript": transcript,
                "transcription_confidence": transcription_result.confidence,
                "is_phishing": aggregated["is_phishing"],
                "overall_confidence": aggregated["overall_confidence"],
                "risk_score": aggregated["risk_score"],
                "warnings": aggregated["warnings"],
                "biometric_match": aggregated.get("biometric_match"),
                "model_breakdown": {
                    "ml_model": ml_result,
                    "rule_based": rule_result,
                    "voice_biometric": {k: v for k, v in biometric_result.items() if k != 'embedding'}  # Exclude embedding from response
                },
                "audio_metadata": {
                    "duration": audio_data["duration_seconds"],
                    "sample_rate": audio_data["sample_rate"],
                    "format": audio_data["original_format"],
                    "speech_ratio": audio_data["features"].get("speech_ratio", 0)
                }
            }
            
            logger.info(f"[{request_id}] Analysis complete - Phishing: {response['is_phishing']}")
            return response
            
        except Exception as e:
            logger.error(f"[{request_id}] Analysis failed: {e}")
            return {
                "request_id": request_id,
                "status": "failed",
                "error": str(e),
                "is_phishing": False,
                "overall_confidence": 0.0
            }
    
    async def _run_ml_model(self, transcript: str) -> Dict:
        """Run ML model prediction"""
        try:
            return self.ml_model_service.predict(transcript)
        except Exception as e:
            logger.error(f"ML model prediction failed: {e}")
            return {"confidence": 0.0, "is_phishing": False, "error": str(e)}
    
    async def _run_rule_based_detection(self, transcript: str) -> Dict:
        """Run rule-based phishing detection"""
        try:
            # Keyword detection
            keywords = self.stt_service.detect_phishing_keywords(transcript)
            
            # Calculate confidence based on keywords
            if keywords:
                confidence = min(sum(keywords.values()) / len(keywords), 1.0)
                warnings = [f"Detected: {keyword} (score: {score:.2f})" 
                           for keyword, score in keywords.items()]
            else:
                confidence = 0.0
                warnings = []
            
            return {
                "confidence": confidence,
                "keywords_detected": keywords,
                "warnings": warnings,
                "model_name": "Rule-based Detection"
            }
            
        except Exception as e:
            logger.error(f"Rule-based detection failed: {e}")
            return {"confidence": 0.0, "warnings": [], "error": str(e)}
    
    async def _run_biometric_check(self, audio_features: Dict, request_id: str) -> Dict:
        """Run voice biometric check"""
        try:
            mfcc_features = audio_features.get("mfcc", [])
            if not mfcc_features:
                return {"is_known_scammer": False, "confidence": 0.0, "error": "No MFCC features"}
            
            return self.biometric_service.analyze_voice(mfcc_features, request_id)
            
        except Exception as e:
            logger.error(f"Biometric check failed: {e}")
            return {"is_known_scammer": False, "confidence": 0.0, "error": str(e)}
    
    def _aggregate_results(self, ml_result: Dict, rule_result: Dict, 
                          biometric_result: Dict, transcription_confidence: float) -> Dict:
        """Aggregate results from all models into final decision"""
        
        warnings = []
        
        # ML Model warnings
        if ml_result.get("is_phishing", False):
            warnings.append({
                "model": "CNN-BiLSTM-Attention",
                "confidence": ml_result.get("confidence", 0.0),
                "severity": "HIGH" if ml_result.get("confidence", 0) > 0.8 else "MEDIUM",
                "message": f"ML model detected phishing with {ml_result.get('confidence', 0):.2f} confidence"
            })
        
        # Rule-based warnings
        for warning in rule_result.get("warnings", []):
            warnings.append({
                "model": "Rule-based",
                "confidence": rule_result.get("confidence", 0.0),
                "severity": "MEDIUM",
                "message": warning
            })
        
        # Biometric warnings
        biometric_match = None
        if biometric_result.get("is_known_scammer", False):
            warnings.append({
                "model": "Voice Biometric",
                "confidence": biometric_result.get("confidence", 0.0),
                "severity": "CRITICAL",
                "message": f"Voice matches known scammer (similarity: {biometric_result.get('similarity_score', 0):.2f})"
            })
            
            biometric_match = {
                "is_known_scammer": True,
                "similarity_score": biometric_result.get("similarity_score", 0.0),
                "confidence": biometric_result.get("confidence", 0.0),
                "scammer_id": biometric_result.get("matched_scammer_id")
            }
        
        # Calculate overall metrics
        model_confidences = [
            ml_result.get("confidence", 0.0),
            rule_result.get("confidence", 0.0),
            biometric_result.get("confidence", 0.0) * 2  # Higher weight for biometric
        ]
        
        overall_confidence = sum(model_confidences) / len(model_confidences)
        
        # Adjust for transcription quality
        if transcription_confidence < 0.5:
            overall_confidence *= 0.8
        
        # Calculate risk score with weights
        risk_score = (
            ml_result.get("confidence", 0.0) * 0.4 +
            rule_result.get("confidence", 0.0) * 0.3 +
            biometric_result.get("confidence", 0.0) * 0.3
        )
        
        # Boost risk if biometric match
        if biometric_result.get("is_known_scammer", False):
            risk_score = max(risk_score, 0.9)
        
        # Determine if phishing (threshold: 0.5)
        is_phishing = (
            risk_score >= 0.5 or 
            biometric_result.get("is_known_scammer", False) or
            ml_result.get("confidence", 0.0) >= 0.7
        )
        
        return {
            "is_phishing": is_phishing,
            "overall_confidence": overall_confidence,
            "risk_score": risk_score,
            "warnings": warnings,
            "biometric_match": biometric_match
        }
    
    async def get_model_status(self) -> Dict:
        """Get status of all detection models"""
        try:
            ml_info = self.ml_model_service.get_model_info()
            biometric_info = self.biometric_service.get_service_info()
            
            return {
                "ml_model": {
                    "status": "operational" if ml_info["loaded"] else "failed",
                    "info": ml_info
                },
                "voice_biometric": {
                    "status": "operational",
                    "info": biometric_info
                },
                "stt_service": {
                    "status": "operational",
                    "model": self.stt_service.model_name
                }
            }
        except Exception as e:
            logger.error(f"Failed to get model status: {e}")
            return {"error": str(e)}