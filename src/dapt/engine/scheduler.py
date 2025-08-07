"""DAPT Training Scheduler

Scheduler for managing multiple DAPT training tasks:
- Queue management for training jobs
- Resource allocation and monitoring
- Parallel training coordination
- Training status tracking
"""

import json
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, Future

class TrainingStatus(Enum):
    """Training job status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

@dataclass
class TrainingJob:
    """Training job data structure"""
    job_id: str
    country_code: str
    config_path: str
    output_dir: str
    priority: int = 1  # 1=high, 2=medium, 3=low
    status: TrainingStatus = TrainingStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress: float = 0.0
    estimated_duration: Optional[float] = None
    actual_duration: Optional[float] = None
    resource_requirements: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.resource_requirements is None:
            self.resource_requirements = {}

class TrainingScheduler:
    """Scheduler for managing DAPT training jobs"""
    
    def __init__(self, global_config: Dict[str, Any], max_concurrent_jobs: int = 2):
        self.global_config = global_config
        self.max_concurrent_jobs = max_concurrent_jobs
        
        # Job management
        self.job_queue: List[TrainingJob] = []
        self.running_jobs: Dict[str, TrainingJob] = {}
        self.completed_jobs: Dict[str, TrainingJob] = {}
        self.job_futures: Dict[str, Future] = {}
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_jobs)
        self.scheduler_thread = None
        self.is_running = False
        self.lock = threading.Lock()
        
        # Callbacks
        self.job_callbacks: Dict[str, List[Callable]] = {
            "on_job_start": [],
            "on_job_complete": [],
            "on_job_fail": [],
            "on_job_cancel": []
        }
        
        # Logging
        self.logger = self._setup_logging()
        
        # State persistence
        self.state_file = Path(global_config["log_dir"]) / "scheduler_state.json"
        self.load_state()
        
        self.logger.info(f"Training Scheduler initialized with max {max_concurrent_jobs} concurrent jobs")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for scheduler"""
        logger = logging.getLogger("dapt_scheduler")
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        log_path = Path(self.global_config["log_dir"]) / "scheduler.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def add_job(self, 
                country_code: str, 
                config_path: str, 
                output_dir: str,
                priority: int = 1,
                resource_requirements: Optional[Dict[str, Any]] = None) -> str:
        """Add a new training job to the queue"""
        
        job_id = f"job_{country_code}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        job = TrainingJob(
            job_id=job_id,
            country_code=country_code,
            config_path=config_path,
            output_dir=output_dir,
            priority=priority,
            resource_requirements=resource_requirements or {}
        )
        
        with self.lock:
            self.job_queue.append(job)
            # Sort by priority (lower number = higher priority)
            self.job_queue.sort(key=lambda x: (x.priority, x.created_at))
        
        self.save_state()
        self.logger.info(f"Added job {job_id} for country {country_code} to queue")
        
        return job_id
    
    def remove_job(self, job_id: str) -> bool:
        """Remove a job from queue or cancel if running"""
        with self.lock:
            # Check if job is in queue
            for i, job in enumerate(self.job_queue):
                if job.job_id == job_id:
                    self.job_queue.pop(i)
                    self.logger.info(f"Removed job {job_id} from queue")
                    self.save_state()
                    return True
            
            # Check if job is running
            if job_id in self.running_jobs:
                return self.cancel_job(job_id)
        
        return False
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        with self.lock:
            if job_id not in self.running_jobs:
                self.logger.warning(f"Job {job_id} is not running")
                return False
            
            # Cancel the future
            if job_id in self.job_futures:
                future = self.job_futures[job_id]
                if not future.done():
                    future.cancel()
                
                del self.job_futures[job_id]
            
            # Update job status
            job = self.running_jobs[job_id]
            job.status = TrainingStatus.CANCELLED
            job.completed_at = datetime.now()
            
            # Move to completed jobs
            self.completed_jobs[job_id] = job
            del self.running_jobs[job_id]
            
            # Trigger callbacks
            self._trigger_callbacks("on_job_cancel", job)
            
            self.save_state()
            self.logger.info(f"Cancelled job {job_id}")
            
            return True
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job"""
        with self.lock:
            # Check queue
            for job in self.job_queue:
                if job.job_id == job_id:
                    return self._job_to_dict(job)
            
            # Check running jobs
            if job_id in self.running_jobs:
                return self._job_to_dict(self.running_jobs[job_id])
            
            # Check completed jobs
            if job_id in self.completed_jobs:
                return self._job_to_dict(self.completed_jobs[job_id])
        
        return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get overall queue status"""
        with self.lock:
            return {
                "queue_length": len(self.job_queue),
                "running_jobs": len(self.running_jobs),
                "completed_jobs": len(self.completed_jobs),
                "max_concurrent": self.max_concurrent_jobs,
                "is_scheduler_running": self.is_running,
                "queued_jobs": [self._job_to_dict(job) for job in self.job_queue],
                "running_jobs_list": [self._job_to_dict(job) for job in self.running_jobs.values()],
                "recent_completed": [self._job_to_dict(job) for job in 
                                   sorted(self.completed_jobs.values(), 
                                         key=lambda x: x.completed_at or datetime.min, 
                                         reverse=True)[:10]]
            }
    
    def start_scheduler(self) -> None:
        """Start the job scheduler"""
        if self.is_running:
            self.logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        self.logger.info("Training scheduler started")
    
    def stop_scheduler(self) -> None:
        """Stop the job scheduler"""
        if not self.is_running:
            self.logger.warning("Scheduler is not running")
            return
        
        self.is_running = False
        
        # Cancel all running jobs
        with self.lock:
            running_job_ids = list(self.running_jobs.keys())
        
        for job_id in running_job_ids:
            self.cancel_job(job_id)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        self.save_state()
        self.logger.info("Training scheduler stopped")
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop"""
        while self.is_running:
            try:
                self._process_queue()
                self._check_running_jobs()
                time.sleep(1)  # Check every second
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {str(e)}")
                time.sleep(5)  # Wait longer on error
    
    def _process_queue(self) -> None:
        """Process jobs in the queue"""
        with self.lock:
            # Check if we can start new jobs
            available_slots = self.max_concurrent_jobs - len(self.running_jobs)
            
            if available_slots <= 0 or not self.job_queue:
                return
            
            # Start jobs up to available slots
            jobs_to_start = self.job_queue[:available_slots]
            
            for job in jobs_to_start:
                if self._can_start_job(job):
                    self._start_job(job)
                    self.job_queue.remove(job)
    
    def _can_start_job(self, job: TrainingJob) -> bool:
        """Check if a job can be started based on resource requirements"""
        # Basic resource checking - can be extended
        required_memory = job.resource_requirements.get("memory_gb", 0)
        required_gpu = job.resource_requirements.get("gpu_count", 0)
        
        # Simple check - in a real implementation, you'd check actual system resources
        return True
    
    def _start_job(self, job: TrainingJob) -> None:
        """Start a training job"""
        job.status = TrainingStatus.RUNNING
        job.started_at = datetime.now()
        
        # Move to running jobs
        self.running_jobs[job.job_id] = job
        
        # Submit to executor
        future = self.executor.submit(self._execute_job, job)
        self.job_futures[job.job_id] = future
        
        # Trigger callbacks
        self._trigger_callbacks("on_job_start", job)
        
        self.logger.info(f"Started job {job.job_id} for country {job.country_code}")
        self.save_state()
    
    def _execute_job(self, job: TrainingJob) -> None:
        """Execute a training job"""
        try:
            from .trainer import DAPTTrainingEngine
            from ..config.manager import ConfigManager
            from ..data.processor import DataProcessor
            
            # Load configuration
            config_manager = ConfigManager(self.global_config)
            config = config_manager.load_config(job.country_code)
            
            if not config:
                raise ValueError(f"Could not load config for country {job.country_code}")
            
            # Initialize data processor
            data_processor = DataProcessor(config, self.global_config)
            
            # Load and prepare data
            train_dataset, val_dataset = data_processor.load_datasets()
            
            # Initialize training engine
            trainer = DAPTTrainingEngine(config, self.global_config)
            
            # Initialize model
            if not trainer.initialize_model():
                raise RuntimeError("Failed to initialize model")
            
            # Prepare datasets
            if not trainer.prepare_datasets(train_dataset, val_dataset):
                raise RuntimeError("Failed to prepare datasets")
            
            # Setup trainer
            if not trainer.setup_trainer(job.output_dir):
                raise RuntimeError("Failed to setup trainer")
            
            # Start training
            results = trainer.train(job.output_dir)
            
            # Update job status
            with self.lock:
                job.status = TrainingStatus.COMPLETED
                job.completed_at = datetime.now()
                job.actual_duration = (job.completed_at - job.started_at).total_seconds()
                job.progress = 100.0
                
                # Move to completed jobs
                self.completed_jobs[job.job_id] = job
                if job.job_id in self.running_jobs:
                    del self.running_jobs[job.job_id]
                if job.job_id in self.job_futures:
                    del self.job_futures[job.job_id]
            
            # Trigger callbacks
            self._trigger_callbacks("on_job_complete", job)
            
            self.logger.info(f"Job {job.job_id} completed successfully")
            
        except Exception as e:
            # Update job status on failure
            with self.lock:
                job.status = TrainingStatus.FAILED
                job.completed_at = datetime.now()
                job.error_message = str(e)
                
                if job.started_at:
                    job.actual_duration = (job.completed_at - job.started_at).total_seconds()
                
                # Move to completed jobs
                self.completed_jobs[job.job_id] = job
                if job.job_id in self.running_jobs:
                    del self.running_jobs[job.job_id]
                if job.job_id in self.job_futures:
                    del self.job_futures[job.job_id]
            
            # Trigger callbacks
            self._trigger_callbacks("on_job_fail", job)
            
            self.logger.error(f"Job {job.job_id} failed: {str(e)}")
        
        finally:
            self.save_state()
    
    def _check_running_jobs(self) -> None:
        """Check status of running jobs"""
        with self.lock:
            completed_jobs = []
            
            for job_id, future in self.job_futures.items():
                if future.done():
                    completed_jobs.append(job_id)
            
            # Clean up completed futures
            for job_id in completed_jobs:
                if job_id in self.job_futures:
                    del self.job_futures[job_id]
    
    def _trigger_callbacks(self, event: str, job: TrainingJob) -> None:
        """Trigger registered callbacks"""
        for callback in self.job_callbacks.get(event, []):
            try:
                callback(job)
            except Exception as e:
                self.logger.error(f"Error in callback for {event}: {str(e)}")
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """Register a callback for job events"""
        if event in self.job_callbacks:
            self.job_callbacks[event].append(callback)
        else:
            self.logger.warning(f"Unknown event type: {event}")
    
    def _job_to_dict(self, job: TrainingJob) -> Dict[str, Any]:
        """Convert job to dictionary"""
        job_dict = asdict(job)
        
        # Convert datetime objects to ISO strings
        for key, value in job_dict.items():
            if isinstance(value, datetime):
                job_dict[key] = value.isoformat()
            elif key == "status":
                job_dict[key] = value.value if hasattr(value, 'value') else str(value)
        
        return job_dict
    
    def save_state(self) -> None:
        """Save scheduler state to file"""
        try:
            state = {
                "queue": [self._job_to_dict(job) for job in self.job_queue],
                "running": [self._job_to_dict(job) for job in self.running_jobs.values()],
                "completed": [self._job_to_dict(job) for job in self.completed_jobs.values()],
                "last_updated": datetime.now().isoformat()
            }
            
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Error saving scheduler state: {str(e)}")
    
    def load_state(self) -> None:
        """Load scheduler state from file"""
        try:
            if not self.state_file.exists():
                return
            
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Restore jobs (only queue and completed, running jobs are lost on restart)
            for job_data in state.get("queue", []):
                job = self._dict_to_job(job_data)
                self.job_queue.append(job)
            
            for job_data in state.get("completed", []):
                job = self._dict_to_job(job_data)
                self.completed_jobs[job.job_id] = job
            
            # Sort queue by priority
            self.job_queue.sort(key=lambda x: (x.priority, x.created_at))
            
            self.logger.info(f"Loaded scheduler state: {len(self.job_queue)} queued, {len(self.completed_jobs)} completed")
            
        except Exception as e:
            self.logger.error(f"Error loading scheduler state: {str(e)}")
    
    def _dict_to_job(self, job_data: Dict[str, Any]) -> TrainingJob:
        """Convert dictionary to job object"""
        # Convert ISO strings back to datetime objects
        for key in ['created_at', 'started_at', 'completed_at']:
            if job_data.get(key):
                job_data[key] = datetime.fromisoformat(job_data[key])
        
        # Convert status string back to enum
        if 'status' in job_data:
            job_data['status'] = TrainingStatus(job_data['status'])
        
        return TrainingJob(**job_data)
    
    def cleanup(self) -> None:
        """Cleanup scheduler resources"""
        self.stop_scheduler()
        self.save_state()
        self.logger.info("Scheduler cleanup completed")