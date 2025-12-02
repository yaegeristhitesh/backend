"""
Model Registry for managing multiple phishing detection models
Supports versioning, A/B testing, and ensemble deployments
"""
import json
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    model_id: str
    model_type: str  # "phishing_detection", "voice_biometric", "stt"
    version: str
    accuracy: float
    training_date: str
    model_path: str
    config_path: str
    status: str  # "active", "deprecated", "testing"
    specialization: str  # "general", "bank_phishing", "irs_scam", "tech_support", "romance_scam"
    performance_metrics: Dict[str, float]
    resource_requirements: Dict[str, Any]

class ModelRegistry:
    """Central registry for all ML models in the system"""
    
    def __init__(self, registry_path: str = "models/registry.json"):
        self.registry_path = Path(registry_path)
        self.models: Dict[str, ModelMetadata] = {}
        self.load_registry()
    
    def load_registry(self):
        """Load model registry from disk"""
        try:
            if self.registry_path.exists():
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                    self.models = {
                        k: ModelMetadata(**v) for k, v in data.items()
                    }
                logger.info(f"Loaded {len(self.models)} models from registry")
            else:
                self._create_default_registry()
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            self._create_default_registry()
    
    def _create_default_registry(self):
        """Create default registry with current model"""
        self.models = {
            "phish_model_v1": ModelMetadata(
                model_id="phish_model_v1",
                model_type="phishing_detection",
                version="1.0.0",
                accuracy=0.88,
                training_date="2024-01-15",
                model_path="models/model_1",
                config_path="models/model_1/model_architecture.json",
                status="active",
                specialization="general",
                performance_metrics={
                    "precision": 0.89,
                    "recall": 0.87,
                    "f1_score": 0.88,
                    "inference_time_ms": 45
                },
                resource_requirements={
                    "memory_mb": 512,
                    "cpu_cores": 1,
                    "gpu_required": False
                }
            )
        }
        self.save_registry()
    
    def register_model(self, metadata: ModelMetadata) -> bool:
        """Register a new model"""
        try:
            self.models[metadata.model_id] = metadata
            self.save_registry()
            logger.info(f"Registered model: {metadata.model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return False
    
    def get_active_models(self, model_type: str = None, specialization: str = None) -> List[ModelMetadata]:
        """Get all active models, optionally filtered"""
        models = [m for m in self.models.values() if m.status == "active"]
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        if specialization:
            models = [m for m in models if m.specialization == specialization]
        
        return models
    
    def get_best_model(self, model_type: str, metric: str = "accuracy") -> Optional[ModelMetadata]:
        """Get the best performing model for a type"""
        candidates = self.get_active_models(model_type)
        if not candidates:
            return None
        
        return max(candidates, key=lambda m: m.performance_metrics.get(metric, 0))
    
    def get_ensemble_models(self, model_type: str) -> List[ModelMetadata]:
        """Get models suitable for ensemble prediction"""
        return [m for m in self.get_active_models(model_type) 
                if m.performance_metrics.get("accuracy", 0) > 0.8]
    
    def save_registry(self):
        """Save registry to disk"""
        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.registry_path, 'w') as f:
                data = {k: asdict(v) for k, v in self.models.items()}
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about registered models"""
        stats = {
            "total_models": len(self.models),
            "active_models": len([m for m in self.models.values() if m.status == "active"]),
            "model_types": {},
            "specializations": {},
            "avg_accuracy": 0.0,
            "resource_usage": {"total_memory_mb": 0, "total_cpu_cores": 0}
        }
        
        for model in self.models.values():
            # Count by type
            stats["model_types"][model.model_type] = stats["model_types"].get(model.model_type, 0) + 1
            
            # Count by specialization
            stats["specializations"][model.specialization] = stats["specializations"].get(model.specialization, 0) + 1
            
            # Resource usage
            if model.status == "active":
                stats["resource_usage"]["total_memory_mb"] += model.resource_requirements.get("memory_mb", 0)
                stats["resource_usage"]["total_cpu_cores"] += model.resource_requirements.get("cpu_cores", 0)
        
        # Calculate average accuracy
        accuracies = [m.accuracy for m in self.models.values() if m.status == "active"]
        stats["avg_accuracy"] = sum(accuracies) / len(accuracies) if accuracies else 0.0
        
        return stats