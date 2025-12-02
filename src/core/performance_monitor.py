"""
Performance monitoring system for tracking model inference metrics,
system resources, and detection accuracy in real-time
"""
import time
import psutil
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class InferenceMetrics:
    request_id: str
    model_id: str
    start_time: float
    end_time: float
    processing_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    prediction_confidence: float
    is_phishing: bool
    audio_duration_seconds: float
    throughput_fps: float  # frames per second processed

@dataclass
class SystemMetrics:
    timestamp: float
    cpu_usage_percent: float
    memory_usage_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    active_requests: int
    queue_size: int

class PerformanceMonitor:
    """Real-time performance monitoring for the phishing detection system"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.inference_history: deque = deque(maxlen=max_history)
        self.system_history: deque = deque(maxlen=max_history)
        self.model_stats: Dict[str, Dict] = defaultdict(lambda: {
            "total_requests": 0,
            "avg_processing_time": 0.0,
            "avg_confidence": 0.0,
            "phishing_detected": 0,
            "accuracy_samples": deque(maxlen=100)
        })
        self.active_requests: Dict[str, float] = {}
        self.start_time = time.time()
        
        # Start background monitoring
        self._monitoring_task = None
        # Don't start monitoring automatically - will be started when needed
    
    def start_monitoring(self):
        """Start background system monitoring"""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitor_system())
    
    async def _monitor_system(self):
        """Background task to collect system metrics"""
        while True:
            try:
                metrics = SystemMetrics(
                    timestamp=time.time(),
                    cpu_usage_percent=psutil.cpu_percent(interval=1),
                    memory_usage_mb=psutil.virtual_memory().used / 1024 / 1024,
                    memory_available_mb=psutil.virtual_memory().available / 1024 / 1024,
                    disk_usage_percent=psutil.disk_usage('/').percent,
                    active_requests=len(self.active_requests),
                    queue_size=0  # Would be actual queue size in production
                )
                self.system_history.append(metrics)
                await asyncio.sleep(5)  # Collect every 5 seconds
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(10)
    
    def start_inference(self, request_id: str, model_id: str) -> float:
        """Mark the start of an inference request"""
        start_time = time.time()
        self.active_requests[request_id] = start_time
        return start_time
    
    def end_inference(self, request_id: str, model_id: str, start_time: float,
                     prediction_confidence: float, is_phishing: bool,
                     audio_duration: float = 0.0) -> InferenceMetrics:
        """Mark the end of an inference request and record metrics"""
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000  # Convert to ms
        
        # Get current system resources
        memory_usage = psutil.virtual_memory().used / 1024 / 1024
        cpu_usage = psutil.cpu_percent()
        
        # Calculate throughput
        throughput = audio_duration / (processing_time / 1000) if processing_time > 0 else 0
        
        metrics = InferenceMetrics(
            request_id=request_id,
            model_id=model_id,
            start_time=start_time,
            end_time=end_time,
            processing_time_ms=processing_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            prediction_confidence=prediction_confidence,
            is_phishing=is_phishing,
            audio_duration_seconds=audio_duration,
            throughput_fps=throughput
        )
        
        # Store metrics
        self.inference_history.append(metrics)
        self._update_model_stats(model_id, metrics)
        
        # Remove from active requests
        self.active_requests.pop(request_id, None)
        
        return metrics
    
    def _update_model_stats(self, model_id: str, metrics: InferenceMetrics):
        """Update running statistics for a model"""
        stats = self.model_stats[model_id]
        stats["total_requests"] += 1
        
        # Update running averages
        n = stats["total_requests"]
        stats["avg_processing_time"] = (
            (stats["avg_processing_time"] * (n-1) + metrics.processing_time_ms) / n
        )
        stats["avg_confidence"] = (
            (stats["avg_confidence"] * (n-1) + metrics.prediction_confidence) / n
        )
        
        if metrics.is_phishing:
            stats["phishing_detected"] += 1
    
    def get_model_performance(self, model_id: str, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance metrics for a specific model"""
        cutoff_time = time.time() - (time_window_minutes * 60)
        recent_metrics = [
            m for m in self.inference_history 
            if m.model_id == model_id and m.start_time >= cutoff_time
        ]
        
        if not recent_metrics:
            return {"error": "No recent data available"}
        
        processing_times = [m.processing_time_ms for m in recent_metrics]
        confidences = [m.prediction_confidence for m in recent_metrics]
        throughputs = [m.throughput_fps for m in recent_metrics if m.throughput_fps > 0]
        
        return {
            "model_id": model_id,
            "time_window_minutes": time_window_minutes,
            "total_requests": len(recent_metrics),
            "phishing_detected": sum(1 for m in recent_metrics if m.is_phishing),
            "avg_processing_time_ms": sum(processing_times) / len(processing_times),
            "min_processing_time_ms": min(processing_times),
            "max_processing_time_ms": max(processing_times),
            "avg_confidence": sum(confidences) / len(confidences),
            "avg_throughput_fps": sum(throughputs) / len(throughputs) if throughputs else 0,
            "requests_per_minute": len(recent_metrics) / time_window_minutes
        }
    
    def get_system_performance(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get system-wide performance metrics"""
        cutoff_time = time.time() - (time_window_minutes * 60)
        recent_metrics = [
            m for m in self.system_history 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {"error": "No recent system data available"}
        
        cpu_usage = [m.cpu_usage_percent for m in recent_metrics]
        memory_usage = [m.memory_usage_mb for m in recent_metrics]
        active_requests = [m.active_requests for m in recent_metrics]
        
        return {
            "time_window_minutes": time_window_minutes,
            "avg_cpu_usage_percent": sum(cpu_usage) / len(cpu_usage),
            "max_cpu_usage_percent": max(cpu_usage),
            "avg_memory_usage_mb": sum(memory_usage) / len(memory_usage),
            "max_memory_usage_mb": max(memory_usage),
            "avg_active_requests": sum(active_requests) / len(active_requests),
            "max_concurrent_requests": max(active_requests),
            "uptime_hours": (time.time() - self.start_time) / 3600
        }
    
    def get_comparative_analysis(self) -> Dict[str, Any]:
        """Compare performance across all models"""
        analysis = {
            "models": {},
            "best_performing": {},
            "resource_efficiency": {}
        }
        
        for model_id in self.model_stats.keys():
            perf = self.get_model_performance(model_id)
            if "error" not in perf:
                analysis["models"][model_id] = perf
        
        if analysis["models"]:
            # Find best performing models
            models = analysis["models"]
            analysis["best_performing"] = {
                "fastest": min(models.items(), key=lambda x: x[1]["avg_processing_time_ms"]),
                "most_confident": max(models.items(), key=lambda x: x[1]["avg_confidence"]),
                "highest_throughput": max(models.items(), key=lambda x: x[1].get("avg_throughput_fps", 0))
            }
        
        return analysis
    
    def export_metrics(self, filepath: str):
        """Export all metrics to JSON file"""
        try:
            data = {
                "inference_history": [asdict(m) for m in self.inference_history],
                "system_history": [asdict(m) for m in self.system_history],
                "model_stats": dict(self.model_stats),
                "export_timestamp": time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Metrics exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")

# Global performance monitor instance - will be initialized when needed
performance_monitor = None

def get_performance_monitor():
    """Get or create the global performance monitor instance"""
    global performance_monitor
    if performance_monitor is None:
        performance_monitor = PerformanceMonitor()
    return performance_monitor