"""DAPT Training Monitor

Real-time monitoring system for DAPT training:
- Training progress tracking
- Performance metrics collection
- Resource usage monitoring
- Alert system for anomalies
"""

import os
import json
import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque
import logging

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

@dataclass
class SystemMetrics:
    """System resource metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_usage_percent: float
    gpu_metrics: Optional[Dict[str, Any]] = None

@dataclass
class TrainingMetrics:
    """Training progress metrics"""
    timestamp: datetime
    training_id: str
    epoch: Optional[int] = None
    step: Optional[int] = None
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    eval_loss: Optional[float] = None
    progress_percent: Optional[float] = None
    estimated_time_remaining: Optional[float] = None

class TrainingMonitor:
    """Monitor for DAPT training processes"""
    
    def __init__(self, global_config: Dict[str, Any], monitoring_interval: int = 30):
        self.global_config = global_config
        self.monitoring_interval = monitoring_interval  # seconds
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Active training sessions
        self.active_trainings: Dict[str, Dict[str, Any]] = {}
        
        # Metrics storage
        self.system_metrics: deque = deque(maxlen=1000)  # Keep last 1000 measurements
        self.training_metrics: Dict[str, deque] = {}  # Per training ID
        
        # Alert system
        self.alert_callbacks: List[Callable] = []
        self.alert_thresholds = {
            "cpu_percent": 90.0,
            "memory_percent": 85.0,
            "disk_usage_percent": 90.0,
            "gpu_memory_percent": 90.0,
            "gpu_temperature": 85.0
        }
        
        # Logging
        self.logger = self._setup_logging()
        
        # Data persistence
        self.metrics_dir = Path(global_config["log_dir"]) / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Training Monitor initialized with {monitoring_interval}s interval")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for monitor"""
        logger = logging.getLogger("dapt_monitor")
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        log_path = Path(self.global_config["log_dir"]) / "monitor.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def start_monitoring(self) -> None:
        """Start the monitoring system"""
        if self.is_monitoring:
            self.logger.warning("Monitor is already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Training monitor started")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring system"""
        if not self.is_monitoring:
            self.logger.warning("Monitor is not running")
            return
        
        self.is_monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        # Save final metrics
        self._save_metrics()
        
        self.logger.info("Training monitor stopped")
    
    def register_training(self, training_id: str, config: Dict[str, Any]) -> None:
        """Register a new training session for monitoring"""
        self.active_trainings[training_id] = {
            "config": config,
            "start_time": datetime.now(),
            "last_update": datetime.now(),
            "status": "active"
        }
        
        # Initialize metrics storage for this training
        self.training_metrics[training_id] = deque(maxlen=1000)
        
        self.logger.info(f"Registered training {training_id} for monitoring")
    
    def unregister_training(self, training_id: str) -> None:
        """Unregister a training session"""
        if training_id in self.active_trainings:
            self.active_trainings[training_id]["status"] = "completed"
            self.active_trainings[training_id]["end_time"] = datetime.now()
            
            # Save final metrics for this training
            self._save_training_metrics(training_id)
            
            self.logger.info(f"Unregistered training {training_id} from monitoring")
    
    def update_training_metrics(self, training_id: str, metrics: Dict[str, Any]) -> None:
        """Update training metrics for a specific training session"""
        if training_id not in self.active_trainings:
            self.logger.warning(f"Training {training_id} not registered for monitoring")
            return
        
        # Create training metrics object
        training_metric = TrainingMetrics(
            timestamp=datetime.now(),
            training_id=training_id,
            epoch=metrics.get("epoch"),
            step=metrics.get("step"),
            loss=metrics.get("train_loss"),
            learning_rate=metrics.get("learning_rate"),
            accuracy=metrics.get("eval_accuracy"),
            f1_score=metrics.get("eval_f1"),
            precision=metrics.get("eval_precision"),
            recall=metrics.get("eval_recall"),
            eval_loss=metrics.get("eval_loss"),
            progress_percent=metrics.get("progress_percent"),
            estimated_time_remaining=metrics.get("estimated_time_remaining")
        )
        
        # Store metrics
        self.training_metrics[training_id].append(training_metric)
        
        # Update last update time
        self.active_trainings[training_id]["last_update"] = datetime.now()
        
        # Check for training anomalies
        self._check_training_anomalies(training_id, training_metric)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        current_metrics = self._collect_system_metrics()
        
        # Calculate averages over last 10 minutes
        recent_metrics = [m for m in self.system_metrics 
                         if (datetime.now() - m.timestamp).total_seconds() < 600]
        
        if recent_metrics:
            avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        else:
            avg_cpu = current_metrics.cpu_percent
            avg_memory = current_metrics.memory_percent
        
        status = {
            "current": asdict(current_metrics),
            "averages_10min": {
                "cpu_percent": avg_cpu,
                "memory_percent": avg_memory
            },
            "active_trainings": len([t for t in self.active_trainings.values() 
                                   if t["status"] == "active"]),
            "total_trainings": len(self.active_trainings),
            "monitoring_since": min([t["start_time"] for t in self.active_trainings.values()]).isoformat() 
                               if self.active_trainings else None
        }
        
        return status
    
    def get_training_status(self, training_id: str) -> Optional[Dict[str, Any]]:
        """Get status for a specific training"""
        if training_id not in self.active_trainings:
            return None
        
        training_info = self.active_trainings[training_id]
        metrics = list(self.training_metrics.get(training_id, []))
        
        # Get latest metrics
        latest_metrics = metrics[-1] if metrics else None
        
        # Calculate training duration
        start_time = training_info["start_time"]
        current_time = datetime.now()
        duration = (current_time - start_time).total_seconds()
        
        status = {
            "training_id": training_id,
            "status": training_info["status"],
            "start_time": start_time.isoformat(),
            "duration_seconds": duration,
            "last_update": training_info["last_update"].isoformat(),
            "latest_metrics": asdict(latest_metrics) if latest_metrics else None,
            "total_metrics_collected": len(metrics)
        }
        
        # Add progress information if available
        if latest_metrics and latest_metrics.progress_percent is not None:
            status["progress_percent"] = latest_metrics.progress_percent
            
            if latest_metrics.estimated_time_remaining:
                status["estimated_time_remaining"] = latest_metrics.estimated_time_remaining
        
        return status
    
    def get_training_history(self, training_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get training metrics history"""
        if training_id not in self.training_metrics:
            return []
        
        metrics = list(self.training_metrics[training_id])[-limit:]
        return [asdict(m) for m in metrics]
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.system_metrics.append(system_metrics)
                
                # Check for system alerts
                self._check_system_alerts(system_metrics)
                
                # Check training health
                self._check_training_health()
                
                # Periodic save
                if len(self.system_metrics) % 10 == 0:  # Save every 10 measurements
                    self._save_metrics()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU metrics if available
        gpu_metrics = None
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_metrics = []
                    for gpu in gpus:
                        gpu_metrics.append({
                            "id": gpu.id,
                            "name": gpu.name,
                            "memory_used": gpu.memoryUsed,
                            "memory_total": gpu.memoryTotal,
                            "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                            "temperature": gpu.temperature,
                            "load": gpu.load * 100
                        })
            except Exception as e:
                self.logger.warning(f"Error collecting GPU metrics: {str(e)}")
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3),
            disk_usage_percent=disk.percent,
            gpu_metrics=gpu_metrics
        )
    
    def _check_system_alerts(self, metrics: SystemMetrics) -> None:
        """Check for system resource alerts"""
        alerts = []
        
        # CPU alert
        if metrics.cpu_percent > self.alert_thresholds["cpu_percent"]:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        # Memory alert
        if metrics.memory_percent > self.alert_thresholds["memory_percent"]:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        # Disk alert
        if metrics.disk_usage_percent > self.alert_thresholds["disk_usage_percent"]:
            alerts.append(f"High disk usage: {metrics.disk_usage_percent:.1f}%")
        
        # GPU alerts
        if metrics.gpu_metrics:
            for gpu in metrics.gpu_metrics:
                if gpu["memory_percent"] > self.alert_thresholds["gpu_memory_percent"]:
                    alerts.append(f"High GPU {gpu['id']} memory usage: {gpu['memory_percent']:.1f}%")
                
                if gpu["temperature"] > self.alert_thresholds["gpu_temperature"]:
                    alerts.append(f"High GPU {gpu['id']} temperature: {gpu['temperature']}°C")
        
        # Trigger alerts
        for alert in alerts:
            self._trigger_alert("system", alert, metrics)
    
    def _check_training_health(self) -> None:
        """Check health of active trainings"""
        current_time = datetime.now()
        
        for training_id, training_info in self.active_trainings.items():
            if training_info["status"] != "active":
                continue
            
            # Check if training has been silent for too long
            last_update = training_info["last_update"]
            silence_duration = (current_time - last_update).total_seconds()
            
            if silence_duration > 300:  # 5 minutes without updates
                alert_msg = f"Training {training_id} has been silent for {silence_duration:.0f} seconds"
                self._trigger_alert("training", alert_msg, training_info)
    
    def _check_training_anomalies(self, training_id: str, metrics: TrainingMetrics) -> None:
        """Check for training anomalies"""
        # Get recent metrics for comparison
        recent_metrics = list(self.training_metrics[training_id])[-10:]  # Last 10 measurements
        
        if len(recent_metrics) < 3:
            return  # Not enough data for anomaly detection
        
        alerts = []
        
        # Check for loss explosion
        if metrics.loss and metrics.loss > 100:
            alerts.append(f"Training {training_id}: Loss explosion detected ({metrics.loss:.4f})")
        
        # Check for NaN values
        if metrics.loss and (metrics.loss != metrics.loss):  # NaN check
            alerts.append(f"Training {training_id}: NaN loss detected")
        
        # Check for learning rate issues
        if metrics.learning_rate and metrics.learning_rate <= 0:
            alerts.append(f"Training {training_id}: Invalid learning rate ({metrics.learning_rate})")
        
        # Check for stagnant training (loss not improving)
        if len(recent_metrics) >= 5:
            recent_losses = [m.loss for m in recent_metrics[-5:] if m.loss is not None]
            if len(recent_losses) >= 3:
                loss_variance = np.var(recent_losses) if len(recent_losses) > 1 else 0
                if loss_variance < 1e-6:  # Very small variance
                    alerts.append(f"Training {training_id}: Loss appears stagnant")
        
        # Trigger alerts
        for alert in alerts:
            self._trigger_alert("training_anomaly", alert, metrics)
    
    def _trigger_alert(self, alert_type: str, message: str, context: Any) -> None:
        """Trigger an alert"""
        alert_data = {
            "type": alert_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "context": context
        }
        
        # Log the alert
        self.logger.warning(f"ALERT [{alert_type}]: {message}")
        
        # Save alert to file
        alert_file = self.metrics_dir / "alerts.jsonl"
        with open(alert_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(alert_data, default=str) + '\n')
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {str(e)}")
    
    def register_alert_callback(self, callback: Callable) -> None:
        """Register a callback for alerts"""
        self.alert_callbacks.append(callback)
    
    def set_alert_threshold(self, metric: str, threshold: float) -> None:
        """Set alert threshold for a metric"""
        if metric in self.alert_thresholds:
            self.alert_thresholds[metric] = threshold
            self.logger.info(f"Set alert threshold for {metric}: {threshold}")
        else:
            self.logger.warning(f"Unknown metric for alert threshold: {metric}")
    
    def _save_metrics(self) -> None:
        """Save metrics to files"""
        try:
            # Save system metrics
            system_file = self.metrics_dir / f"system_metrics_{datetime.now().strftime('%Y%m%d')}.jsonl"
            
            # Only save recent metrics to avoid huge files
            recent_system_metrics = [m for m in self.system_metrics 
                                   if (datetime.now() - m.timestamp).total_seconds() < 3600]  # Last hour
            
            with open(system_file, 'a', encoding='utf-8') as f:
                for metric in recent_system_metrics:
                    f.write(json.dumps(asdict(metric), default=str) + '\n')
            
            # Save training metrics
            for training_id in self.training_metrics:
                self._save_training_metrics(training_id)
                
        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")
    
    def _save_training_metrics(self, training_id: str) -> None:
        """Save metrics for a specific training"""
        try:
            training_file = self.metrics_dir / f"training_{training_id}.jsonl"
            
            with open(training_file, 'w', encoding='utf-8') as f:
                for metric in self.training_metrics[training_id]:
                    f.write(json.dumps(asdict(metric), default=str) + '\n')
                    
        except Exception as e:
            self.logger.error(f"Error saving training metrics for {training_id}: {str(e)}")
    
    def cleanup(self) -> None:
        """Cleanup monitor resources"""
        self.stop_monitoring()
        self.logger.info("Monitor cleanup completed")