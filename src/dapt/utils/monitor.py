"""DAPT Status Monitoring System

Comprehensive monitoring system for DAPT training:
- Training status tracking
- System resource monitoring
- Real-time metrics collection
- Alert system for anomalies
- Performance bottleneck detection
"""

import os
import sys
import time
import psutil
import threading
import json
from typing import Dict, Any, Optional, List, Callable, Union
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import queue
import subprocess
from contextlib import contextmanager

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class TrainingStatus(Enum):
    """Training status enumeration"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    TRAINING = "training"
    VALIDATING = "validating"
    EVALUATING = "evaluating"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    STOPPED = "stopped"

class AlertLevel(Enum):
    """Alert level enumeration"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class SystemMetrics:
    """System resource metrics"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_sent_mb: float
    network_recv_mb: float
    load_average: List[float]
    process_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class GPUMetrics:
    """GPU metrics"""
    timestamp: str
    gpu_id: int
    gpu_name: str
    gpu_utilization: float
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    temperature: float
    power_draw: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class TrainingMetrics:
    """Training-specific metrics"""
    timestamp: str
    epoch: int
    step: int
    loss: float
    learning_rate: float
    batch_size: int
    samples_per_second: float
    eta_seconds: Optional[float]
    validation_loss: Optional[float]
    validation_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class Alert:
    """System alert"""
    timestamp: str
    level: AlertLevel
    category: str
    message: str
    details: Dict[str, Any]
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class MetricsCollector:
    """Collect system and training metrics"""
    
    def __init__(self, collection_interval: float = 5.0):
        self.collection_interval = collection_interval
        self.running = False
        self.thread = None
        
        # Metrics storage
        self.system_metrics = deque(maxlen=1000)
        self.gpu_metrics = deque(maxlen=1000)
        self.training_metrics = deque(maxlen=1000)
        
        # Network baseline
        self.network_baseline = None
        
        # Initialize baseline
        self._initialize_baseline()
    
    def _initialize_baseline(self) -> None:
        """Initialize baseline metrics"""
        try:
            net_io = psutil.net_io_counters()
            self.network_baseline = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'timestamp': time.time()
            }
        except Exception:
            self.network_baseline = None
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network I/O
            net_sent_mb = 0.0
            net_recv_mb = 0.0
            
            if self.network_baseline:
                try:
                    net_io = psutil.net_io_counters()
                    current_time = time.time()
                    time_diff = current_time - self.network_baseline['timestamp']
                    
                    if time_diff > 0:
                        sent_diff = net_io.bytes_sent - self.network_baseline['bytes_sent']
                        recv_diff = net_io.bytes_recv - self.network_baseline['bytes_recv']
                        
                        net_sent_mb = (sent_diff / time_diff) / (1024 * 1024)  # MB/s
                        net_recv_mb = (recv_diff / time_diff) / (1024 * 1024)  # MB/s
                        
                        # Update baseline
                        self.network_baseline = {
                            'bytes_sent': net_io.bytes_sent,
                            'bytes_recv': net_io.bytes_recv,
                            'timestamp': current_time
                        }
                except Exception:
                    pass
            
            # Load average
            try:
                load_avg = list(os.getloadavg())
            except (OSError, AttributeError):
                load_avg = [0.0, 0.0, 0.0]
            
            # Process count
            process_count = len(psutil.pids())
            
            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3),
                memory_total_gb=memory.total / (1024**3),
                disk_usage_percent=disk.percent,
                disk_free_gb=disk.free / (1024**3),
                network_sent_mb=net_sent_mb,
                network_recv_mb=net_recv_mb,
                load_average=load_avg,
                process_count=process_count
            )
            
        except Exception as e:
            # Return default metrics on error
            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_gb=0.0,
                memory_total_gb=0.0,
                disk_usage_percent=0.0,
                disk_free_gb=0.0,
                network_sent_mb=0.0,
                network_recv_mb=0.0,
                load_average=[0.0, 0.0, 0.0],
                process_count=0
            )
    
    def collect_gpu_metrics(self) -> List[GPUMetrics]:
        """Collect GPU metrics"""
        gpu_metrics = []
        
        if not GPU_AVAILABLE:
            return gpu_metrics
        
        try:
            gpus = GPUtil.getGPUs()
            
            for gpu in gpus:
                metrics = GPUMetrics(
                    timestamp=datetime.now().isoformat(),
                    gpu_id=gpu.id,
                    gpu_name=gpu.name,
                    gpu_utilization=gpu.load * 100,
                    memory_used_mb=gpu.memoryUsed,
                    memory_total_mb=gpu.memoryTotal,
                    memory_percent=(gpu.memoryUsed / gpu.memoryTotal) * 100,
                    temperature=gpu.temperature,
                    power_draw=getattr(gpu, 'powerDraw', 0.0)
                )
                gpu_metrics.append(metrics)
                
        except Exception as e:
            pass
        
        # Also try PyTorch GPU info if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    memory_used = torch.cuda.memory_allocated(i) / (1024**2)  # MB
                    memory_total = props.total_memory / (1024**2)  # MB
                    
                    # Check if we already have this GPU from GPUtil
                    existing = any(gpu.gpu_id == i for gpu in gpu_metrics)
                    
                    if not existing:
                        metrics = GPUMetrics(
                            timestamp=datetime.now().isoformat(),
                            gpu_id=i,
                            gpu_name=props.name,
                            gpu_utilization=0.0,  # Not available from PyTorch
                            memory_used_mb=memory_used,
                            memory_total_mb=memory_total,
                            memory_percent=(memory_used / memory_total) * 100,
                            temperature=0.0,  # Not available from PyTorch
                            power_draw=0.0   # Not available from PyTorch
                        )
                        gpu_metrics.append(metrics)
                        
            except Exception as e:
                pass
        
        return gpu_metrics
    
    def start_collection(self) -> None:
        """Start metrics collection in background thread"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.thread.start()
    
    def stop_collection(self) -> None:
        """Stop metrics collection"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
    
    def _collection_loop(self) -> None:
        """Main collection loop"""
        while self.running:
            try:
                # Collect system metrics
                sys_metrics = self.collect_system_metrics()
                self.system_metrics.append(sys_metrics)
                
                # Collect GPU metrics
                gpu_metrics = self.collect_gpu_metrics()
                for gpu_metric in gpu_metrics:
                    self.gpu_metrics.append(gpu_metric)
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                time.sleep(self.collection_interval)
    
    def add_training_metrics(self, metrics: TrainingMetrics) -> None:
        """Add training metrics"""
        self.training_metrics.append(metrics)
    
    def get_recent_metrics(self, minutes: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Get metrics from the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        recent_system = []
        recent_gpu = []
        recent_training = []
        
        # Filter system metrics
        for metric in self.system_metrics:
            try:
                metric_time = datetime.fromisoformat(metric.timestamp)
                if metric_time >= cutoff_time:
                    recent_system.append(metric.to_dict())
            except Exception:
                continue
        
        # Filter GPU metrics
        for metric in self.gpu_metrics:
            try:
                metric_time = datetime.fromisoformat(metric.timestamp)
                if metric_time >= cutoff_time:
                    recent_gpu.append(metric.to_dict())
            except Exception:
                continue
        
        # Filter training metrics
        for metric in self.training_metrics:
            try:
                metric_time = datetime.fromisoformat(metric.timestamp)
                if metric_time >= cutoff_time:
                    recent_training.append(metric.to_dict())
            except Exception:
                continue
        
        return {
            'system': recent_system,
            'gpu': recent_gpu,
            'training': recent_training
        }

class AlertManager:
    """Manage system alerts and notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alerts = deque(maxlen=1000)
        self.alert_callbacks = []
        
        # Alert thresholds
        self.thresholds = config.get("alert_thresholds", {
            "cpu_percent": 90.0,
            "memory_percent": 85.0,
            "disk_usage_percent": 90.0,
            "gpu_memory_percent": 90.0,
            "gpu_temperature": 85.0
        })
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def create_alert(self, level: AlertLevel, category: str, message: str, details: Dict[str, Any] = None) -> Alert:
        """Create and process new alert"""
        alert = Alert(
            timestamp=datetime.now().isoformat(),
            level=level,
            category=category,
            message=message,
            details=details or {},
            resolved=False
        )
        
        self.alerts.append(alert)
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception:
                pass
        
        return alert
    
    def check_system_alerts(self, metrics: SystemMetrics) -> List[Alert]:
        """Check system metrics for alert conditions"""
        alerts = []
        
        # CPU usage alert
        if metrics.cpu_percent > self.thresholds["cpu_percent"]:
            alert = self.create_alert(
                AlertLevel.WARNING,
                "system",
                f"High CPU usage: {metrics.cpu_percent:.1f}%",
                {"cpu_percent": metrics.cpu_percent, "threshold": self.thresholds["cpu_percent"]}
            )
            alerts.append(alert)
        
        # Memory usage alert
        if metrics.memory_percent > self.thresholds["memory_percent"]:
            alert = self.create_alert(
                AlertLevel.WARNING,
                "system",
                f"High memory usage: {metrics.memory_percent:.1f}%",
                {"memory_percent": metrics.memory_percent, "threshold": self.thresholds["memory_percent"]}
            )
            alerts.append(alert)
        
        # Disk usage alert
        if metrics.disk_usage_percent > self.thresholds["disk_usage_percent"]:
            alert = self.create_alert(
                AlertLevel.ERROR,
                "system",
                f"High disk usage: {metrics.disk_usage_percent:.1f}%",
                {"disk_usage_percent": metrics.disk_usage_percent, "threshold": self.thresholds["disk_usage_percent"]}
            )
            alerts.append(alert)
        
        return alerts
    
    def check_gpu_alerts(self, gpu_metrics: List[GPUMetrics]) -> List[Alert]:
        """Check GPU metrics for alert conditions"""
        alerts = []
        
        for gpu_metric in gpu_metrics:
            # GPU memory alert
            if gpu_metric.memory_percent > self.thresholds["gpu_memory_percent"]:
                alert = self.create_alert(
                    AlertLevel.WARNING,
                    "gpu",
                    f"High GPU {gpu_metric.gpu_id} memory usage: {gpu_metric.memory_percent:.1f}%",
                    {
                        "gpu_id": gpu_metric.gpu_id,
                        "memory_percent": gpu_metric.memory_percent,
                        "threshold": self.thresholds["gpu_memory_percent"]
                    }
                )
                alerts.append(alert)
            
            # GPU temperature alert
            if gpu_metric.temperature > self.thresholds["gpu_temperature"]:
                alert = self.create_alert(
                    AlertLevel.ERROR,
                    "gpu",
                    f"High GPU {gpu_metric.gpu_id} temperature: {gpu_metric.temperature:.1f}°C",
                    {
                        "gpu_id": gpu_metric.gpu_id,
                        "temperature": gpu_metric.temperature,
                        "threshold": self.thresholds["gpu_temperature"]
                    }
                )
                alerts.append(alert)
        
        return alerts
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all unresolved alerts"""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, alert_timestamp: str) -> bool:
        """Mark alert as resolved"""
        for alert in self.alerts:
            if alert.timestamp == alert_timestamp:
                alert.resolved = True
                return True
        return False

class TrainingMonitor:
    """Main training monitoring system"""
    
    def __init__(self, config: Dict[str, Any], global_config: Dict[str, Any]):
        self.config = config
        self.global_config = global_config
        self.country_code = config["country"]["code"]
        
        # Initialize components
        self.metrics_collector = MetricsCollector(
            collection_interval=config.get("monitoring", {}).get("collection_interval", 5.0)
        )
        self.alert_manager = AlertManager(config.get("monitoring", {}))
        
        # Training state
        self.training_status = TrainingStatus.IDLE
        self.training_start_time = None
        self.current_epoch = 0
        self.current_step = 0
        self.total_steps = 0
        
        # Status file
        self.status_file = Path(global_config["log_dir"]) / self.country_code / "status.json"
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup alert callbacks
        self.alert_manager.add_alert_callback(self._handle_alert)
        
        # Start monitoring
        self.start_monitoring()
    
    def _handle_alert(self, alert: Alert) -> None:
        """Handle alert notifications"""
        # Log alert (assuming logger is available)
        try:
            from .logger import get_logger
            logger = get_logger("monitor")
            logger.warning(f"Alert [{alert.level.value.upper()}] {alert.category}: {alert.message}")
        except Exception:
            print(f"Alert [{alert.level.value.upper()}] {alert.category}: {alert.message}")
    
    def start_monitoring(self) -> None:
        """Start monitoring system"""
        self.metrics_collector.start_collection()
        self.update_status()
    
    def stop_monitoring(self) -> None:
        """Stop monitoring system"""
        self.metrics_collector.stop_collection()
    
    def set_training_status(self, status: TrainingStatus, details: Dict[str, Any] = None) -> None:
        """Update training status"""
        self.training_status = status
        
        if status == TrainingStatus.TRAINING and self.training_start_time is None:
            self.training_start_time = datetime.now()
        elif status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.STOPPED]:
            self.training_start_time = None
        
        self.update_status(details)
    
    def update_training_progress(self, epoch: int, step: int, total_steps: int, metrics: Dict[str, float] = None) -> None:
        """Update training progress"""
        self.current_epoch = epoch
        self.current_step = step
        self.total_steps = total_steps
        
        # Add training metrics
        if metrics:
            training_metrics = TrainingMetrics(
                timestamp=datetime.now().isoformat(),
                epoch=epoch,
                step=step,
                loss=metrics.get('loss', 0.0),
                learning_rate=metrics.get('learning_rate', 0.0),
                batch_size=metrics.get('batch_size', 0),
                samples_per_second=metrics.get('samples_per_second', 0.0),
                eta_seconds=self._calculate_eta(),
                validation_loss=metrics.get('validation_loss'),
                validation_metrics={k: v for k, v in metrics.items() if k.startswith('val_')}
            )
            self.metrics_collector.add_training_metrics(training_metrics)
        
        self.update_status()
    
    def _calculate_eta(self) -> Optional[float]:
        """Calculate estimated time to completion"""
        if not self.training_start_time or self.total_steps == 0 or self.current_step == 0:
            return None
        
        elapsed_time = (datetime.now() - self.training_start_time).total_seconds()
        steps_per_second = self.current_step / elapsed_time
        
        if steps_per_second > 0:
            remaining_steps = self.total_steps - self.current_step
            return remaining_steps / steps_per_second
        
        return None
    
    def update_status(self, details: Dict[str, Any] = None) -> None:
        """Update status file"""
        try:
            # Get recent metrics
            recent_metrics = self.metrics_collector.get_recent_metrics(minutes=5)
            
            # Calculate progress percentage
            progress_percent = 0.0
            if self.total_steps > 0:
                progress_percent = (self.current_step / self.total_steps) * 100
            
            # Get latest system metrics
            latest_system = None
            if recent_metrics['system']:
                latest_system = recent_metrics['system'][-1]
            
            # Get latest GPU metrics
            latest_gpu = None
            if recent_metrics['gpu']:
                latest_gpu = recent_metrics['gpu'][-1]
            
            status_data = {
                'timestamp': datetime.now().isoformat(),
                'country_code': self.country_code,
                'training_status': self.training_status.value,
                'training_start_time': self.training_start_time.isoformat() if self.training_start_time else None,
                'current_epoch': self.current_epoch,
                'current_step': self.current_step,
                'total_steps': self.total_steps,
                'progress_percent': progress_percent,
                'eta_seconds': self._calculate_eta(),
                'system_metrics': latest_system,
                'gpu_metrics': latest_gpu,
                'active_alerts': len(self.alert_manager.get_active_alerts()),
                'details': details or {}
            }
            
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(status_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        try:
            if self.status_file.exists():
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        
        return {
            'timestamp': datetime.now().isoformat(),
            'country_code': self.country_code,
            'training_status': self.training_status.value,
            'error': 'Status file not available'
        }
    
    def get_metrics_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """Get metrics summary"""
        recent_metrics = self.metrics_collector.get_recent_metrics(minutes)
        
        summary = {
            'time_period_minutes': minutes,
            'system_metrics_count': len(recent_metrics['system']),
            'gpu_metrics_count': len(recent_metrics['gpu']),
            'training_metrics_count': len(recent_metrics['training']),
            'active_alerts': len(self.alert_manager.get_active_alerts())
        }
        
        # Calculate averages for system metrics
        if recent_metrics['system']:
            cpu_avg = sum(m['cpu_percent'] for m in recent_metrics['system']) / len(recent_metrics['system'])
            memory_avg = sum(m['memory_percent'] for m in recent_metrics['system']) / len(recent_metrics['system'])
            
            summary['system_averages'] = {
                'cpu_percent': cpu_avg,
                'memory_percent': memory_avg
            }
        
        # Calculate averages for GPU metrics
        if recent_metrics['gpu']:
            gpu_util_avg = sum(m['gpu_utilization'] for m in recent_metrics['gpu']) / len(recent_metrics['gpu'])
            gpu_memory_avg = sum(m['memory_percent'] for m in recent_metrics['gpu']) / len(recent_metrics['gpu'])
            
            summary['gpu_averages'] = {
                'gpu_utilization': gpu_util_avg,
                'memory_percent': gpu_memory_avg
            }
        
        return summary
    
    def check_health(self) -> Dict[str, Any]:
        """Perform health check"""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        try:
            # Check system metrics
            sys_metrics = self.metrics_collector.collect_system_metrics()
            
            # CPU check
            if sys_metrics.cpu_percent > 95:
                health_status['checks']['cpu'] = 'critical'
                health_status['overall_status'] = 'unhealthy'
            elif sys_metrics.cpu_percent > 80:
                health_status['checks']['cpu'] = 'warning'
            else:
                health_status['checks']['cpu'] = 'healthy'
            
            # Memory check
            if sys_metrics.memory_percent > 90:
                health_status['checks']['memory'] = 'critical'
                health_status['overall_status'] = 'unhealthy'
            elif sys_metrics.memory_percent > 75:
                health_status['checks']['memory'] = 'warning'
            else:
                health_status['checks']['memory'] = 'healthy'
            
            # Disk check
            if sys_metrics.disk_usage_percent > 95:
                health_status['checks']['disk'] = 'critical'
                health_status['overall_status'] = 'unhealthy'
            elif sys_metrics.disk_usage_percent > 85:
                health_status['checks']['disk'] = 'warning'
            else:
                health_status['checks']['disk'] = 'healthy'
            
            # GPU check
            gpu_metrics = self.metrics_collector.collect_gpu_metrics()
            if gpu_metrics:
                gpu_healthy = True
                for gpu in gpu_metrics:
                    if gpu.temperature > 90 or gpu.memory_percent > 95:
                        health_status['checks']['gpu'] = 'critical'
                        health_status['overall_status'] = 'unhealthy'
                        gpu_healthy = False
                        break
                    elif gpu.temperature > 80 or gpu.memory_percent > 85:
                        health_status['checks']['gpu'] = 'warning'
                        gpu_healthy = False
                
                if gpu_healthy:
                    health_status['checks']['gpu'] = 'healthy'
            else:
                health_status['checks']['gpu'] = 'not_available'
            
            # Training status check
            if self.training_status == TrainingStatus.FAILED:
                health_status['checks']['training'] = 'failed'
                health_status['overall_status'] = 'unhealthy'
            elif self.training_status in [TrainingStatus.TRAINING, TrainingStatus.VALIDATING]:
                health_status['checks']['training'] = 'active'
            else:
                health_status['checks']['training'] = 'idle'
            
        except Exception as e:
            health_status['overall_status'] = 'error'
            health_status['error'] = str(e)
        
        return health_status
    
    @contextmanager
    def training_session(self, total_steps: int):
        """Context manager for training session"""
        self.total_steps = total_steps
        self.set_training_status(TrainingStatus.TRAINING)
        
        try:
            yield self
        except Exception as e:
            self.set_training_status(TrainingStatus.FAILED, {'error': str(e)})
            raise
        else:
            self.set_training_status(TrainingStatus.COMPLETED)
        finally:
            self.training_start_time = None
    
    def close(self) -> None:
        """Close monitoring system"""
        self.stop_monitoring()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Global monitor instance
_global_monitor = None

def get_monitor() -> Optional[TrainingMonitor]:
    """Get global monitor instance"""
    return _global_monitor

def setup_global_monitor(config: Dict[str, Any], global_config: Dict[str, Any]) -> TrainingMonitor:
    """Setup global monitor instance"""
    global _global_monitor
    _global_monitor = TrainingMonitor(config, global_config)
    return _global_monitor

def log_training_step(epoch: int, step: int, total_steps: int, metrics: Dict[str, float]) -> None:
    """Convenience function for logging training step"""
    monitor = get_monitor()
    if monitor:
        monitor.update_training_progress(epoch, step, total_steps, metrics)

def set_training_status(status: TrainingStatus, details: Dict[str, Any] = None) -> None:
    """Convenience function for setting training status"""
    monitor = get_monitor()
    if monitor:
        monitor.set_training_status(status, details)