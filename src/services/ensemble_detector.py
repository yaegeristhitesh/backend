"""
Enhanced ensemble detector supporting multiple specialized models
running in parallel for improved accuracy and specialized detection
"""
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from .ml_model_service import MLModelService
from ..core.model_registry import ModelRegistry, ModelMetadata
from ..core.performance_monitor import get_performance_monitor

logger = logging.getLogger(__name__)

class SpecializedModelService(MLModelService):
    """Specialized model service for different scam types"""
    
    def __init__(self, model_metadata: ModelMetadata):
        self.metadata = model_metadata
        super().__init__(model_metadata.model_path)
        self.specialization = model_metadata.specialization
        self.model_id = model_metadata.model_id
    
    async def predict_async(self, text: str, request_id: str) -> Dict:
        """Async wrapper for prediction with performance monitoring"""
        monitor = get_performance_monitor()
        start_time = monitor.start_inference(request_id, self.model_id)
        
        try:
            # Run prediction in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(executor, self.predict, text)
            
            # Record performance metrics
            monitor = get_performance_monitor()
            monitor.end_inference(
                request_id, self.model_id, start_time,
                result.get("confidence", 0.0), result.get("is_phishing", False)
            )
            
            # Add model metadata to result
            result.update({
                "model_id": self.model_id,
                "specialization": self.specialization,
                "model_version": self.metadata.version
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for {self.model_id}: {e}")
            monitor = get_performance_monitor()
            monitor.end_inference(
                request_id, self.model_id, start_time, 0.0, False
            )
            return {
                "model_id": self.model_id,
                "specialization": self.specialization,
                "is_phishing": False,
                "confidence": 0.0,
                "error": str(e)
            }

class EnsembleDetector:
    """
    Advanced ensemble detector that runs multiple specialized models in parallel
    and combines their predictions using weighted voting and confidence scoring
    """
    
    def __init__(self):
        self.registry = ModelRegistry()
        self.models: Dict[str, SpecializedModelService] = {}
        self.ensemble_weights: Dict[str, float] = {}
        self.load_models()
    
    def load_models(self):
        """Load all active phishing detection models"""
        active_models = self.registry.get_active_models("phishing_detection")
        
        for model_metadata in active_models:
            try:
                model_service = SpecializedModelService(model_metadata)
                self.models[model_metadata.model_id] = model_service
                
                # Set ensemble weight based on accuracy
                self.ensemble_weights[model_metadata.model_id] = model_metadata.accuracy
                
                logger.info(f"Loaded model: {model_metadata.model_id} ({model_metadata.specialization})")
                
            except Exception as e:
                logger.error(f"Failed to load model {model_metadata.model_id}: {e}")
        
        # Normalize weights
        total_weight = sum(self.ensemble_weights.values())
        if total_weight > 0:
            self.ensemble_weights = {
                k: v / total_weight for k, v in self.ensemble_weights.items()
            }
        
        logger.info(f"Loaded {len(self.models)} models for ensemble prediction")
    
    async def predict_ensemble(self, text: str, request_id: str) -> Dict:
        """
        Run ensemble prediction using all available models in parallel
        """
        if not self.models:
            return {
                "is_phishing": False,
                "confidence": 0.0,
                "error": "No models available",
                "ensemble_results": []
            }
        
        start_time = time.time()
        
        # Create prediction tasks for all models
        tasks = []
        for model_id, model_service in self.models.items():
            task = asyncio.create_task(
                model_service.predict_async(text, f"{request_id}-{model_id}")
            )
            tasks.append((model_id, task))
        
        # Wait for all predictions
        results = []
        for model_id, task in tasks:
            try:
                result = await task
                results.append(result)
            except Exception as e:
                logger.error(f"Model {model_id} failed: {e}")
                results.append({
                    "model_id": model_id,
                    "is_phishing": False,
                    "confidence": 0.0,
                    "error": str(e)
                })
        
        # Combine predictions using ensemble logic
        ensemble_result = self._combine_predictions(results, text)
        
        # Add timing information
        processing_time = (time.time() - start_time) * 1000
        ensemble_result["processing_time_ms"] = processing_time
        ensemble_result["models_used"] = len(results)
        ensemble_result["ensemble_results"] = results
        
        return ensemble_result
    
    def _combine_predictions(self, results: List[Dict], text: str) -> Dict:
        """
        Combine multiple model predictions using weighted ensemble voting
        """
        if not results:
            return {"is_phishing": False, "confidence": 0.0, "method": "no_results"}
        
        # Filter out failed predictions
        valid_results = [r for r in results if "error" not in r]
        
        if not valid_results:
            return {
                "is_phishing": False,
                "confidence": 0.0,
                "method": "all_failed",
                "failed_models": len(results)
            }
        
        # Method 1: Weighted Average
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for result in valid_results:
            model_id = result.get("model_id", "unknown")
            weight = self.ensemble_weights.get(model_id, 0.1)
            confidence = result.get("confidence", 0.0)
            
            weighted_confidence += confidence * weight
            total_weight += weight
        
        avg_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0
        
        # Method 2: Majority Voting with Confidence Threshold
        high_confidence_predictions = [
            r for r in valid_results 
            if r.get("confidence", 0.0) > 0.7 and r.get("is_phishing", False)
        ]
        
        # Method 3: Specialization-based Decision
        specialized_predictions = self._get_specialized_predictions(valid_results, text)
        
        # Final decision logic
        is_phishing = False
        final_confidence = avg_confidence
        decision_method = "weighted_average"
        
        # High confidence threshold
        if avg_confidence > 0.8:
            is_phishing = True
            decision_method = "high_confidence_ensemble"
        
        # Majority of high-confidence models agree
        elif len(high_confidence_predictions) >= 2:
            is_phishing = True
            final_confidence = max(avg_confidence, 0.85)
            decision_method = "majority_high_confidence"
        
        # Specialized model detected with high confidence
        elif specialized_predictions["max_confidence"] > 0.75:
            is_phishing = True
            final_confidence = specialized_predictions["max_confidence"]
            decision_method = f"specialized_{specialized_predictions['best_specialization']}"
        
        # Conservative threshold for general ensemble
        elif avg_confidence > 0.6:
            is_phishing = True
            decision_method = "conservative_ensemble"
        
        return {
            "is_phishing": is_phishing,
            "confidence": final_confidence,
            "method": decision_method,
            "ensemble_confidence": avg_confidence,
            "high_confidence_votes": len(high_confidence_predictions),
            "specialized_detection": specialized_predictions,
            "model_agreement": self._calculate_agreement(valid_results)
        }
    
    def _get_specialized_predictions(self, results: List[Dict], text: str) -> Dict:
        """Analyze predictions from specialized models"""
        specialized = {
            "detections": {},
            "max_confidence": 0.0,
            "best_specialization": "general"
        }
        
        for result in results:
            specialization = result.get("specialization", "general")
            confidence = result.get("confidence", 0.0)
            is_phishing = result.get("is_phishing", False)
            
            if is_phishing and confidence > specialized["max_confidence"]:
                specialized["max_confidence"] = confidence
                specialized["best_specialization"] = specialization
            
            specialized["detections"][specialization] = {
                "confidence": confidence,
                "is_phishing": is_phishing
            }
        
        return specialized
    
    def _calculate_agreement(self, results: List[Dict]) -> float:
        """Calculate agreement level between models"""
        if len(results) < 2:
            return 1.0
        
        phishing_votes = sum(1 for r in results if r.get("is_phishing", False))
        total_votes = len(results)
        
        # Agreement is how close we are to unanimous decision
        agreement = max(phishing_votes, total_votes - phishing_votes) / total_votes
        return agreement
    
    async def get_model_status(self) -> Dict:
        """Get status of all ensemble models"""
        status = {
            "total_models": len(self.models),
            "active_models": len([m for m in self.models.values() if m.loaded]),
            "models": {},
            "ensemble_weights": self.ensemble_weights,
            "registry_stats": self.registry.get_registry_stats()
        }
        
        for model_id, model_service in self.models.items():
            model_info = model_service.get_model_info()
            model_info["weight"] = self.ensemble_weights.get(model_id, 0.0)
            model_info["specialization"] = model_service.specialization
            status["models"][model_id] = model_info
        
        return status
    
    def add_model(self, model_metadata: ModelMetadata) -> bool:
        """Dynamically add a new model to the ensemble"""
        try:
            # Register in registry
            self.registry.register_model(model_metadata)
            
            # Load model service
            model_service = SpecializedModelService(model_metadata)
            self.models[model_metadata.model_id] = model_service
            
            # Update ensemble weights
            self.ensemble_weights[model_metadata.model_id] = model_metadata.accuracy
            
            # Renormalize weights
            total_weight = sum(self.ensemble_weights.values())
            self.ensemble_weights = {
                k: v / total_weight for k, v in self.ensemble_weights.items()
            }
            
            logger.info(f"Added model {model_metadata.model_id} to ensemble")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add model to ensemble: {e}")
            return False