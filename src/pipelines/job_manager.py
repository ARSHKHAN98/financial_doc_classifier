"""
Job management and state tracking.

In-memory job store designed to be easily swapped with Redis.
"""

import time
import uuid
from enum import Enum
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from threading import Lock

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class JobState(str, Enum):
    """Job processing states."""
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


@dataclass
class JobResult:
    """Job result data."""
    job_id: str
    state: JobState
    created_at: datetime
    updated_at: datetime
    filename: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        data = asdict(self)
        data["state"] = self.state.value
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data


class JobStore:
    """
    In-memory job store with expiration.
    
    Designed to be easily replaceable with Redis for production use.
    """
    
    def __init__(self, cleanup_after_hours: int = 24):
        """
        Initialize job store.
        
        Args:
            cleanup_after_hours: Hours after which completed jobs are removed
        """
        self._store: Dict[str, JobResult] = {}
        self._lock = Lock()
        self._cleanup_after = timedelta(hours=cleanup_after_hours)
    
    def create_job(self, filename: Optional[str] = None) -> str:
        """
        Create a new job.
        
        Args:
            filename: Optional filename for the document
            
        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        job = JobResult(
            job_id=job_id,
            state=JobState.QUEUED,
            created_at=now,
            updated_at=now,
            filename=filename
        )
        
        with self._lock:
            self._store[job_id] = job
        
        logger.info(f"Created job {job_id}")
        return job_id
    
    def get_job(self, job_id: str) -> Optional[JobResult]:
        """
        Get job by ID.
        
        Args:
            job_id: Job ID
            
        Returns:
            JobResult or None if not found
        """
        with self._lock:
            return self._store.get(job_id)
    
    def update_job(
        self,
        job_id: str,
        state: Optional[JobState] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        processing_time_ms: Optional[int] = None
    ) -> bool:
        """
        Update job state and data.
        
        Args:
            job_id: Job ID
            state: New state
            result: Result data
            error: Error message
            processing_time_ms: Processing time in milliseconds
            
        Returns:
            True if updated, False if job not found
        """
        with self._lock:
            job = self._store.get(job_id)
            if not job:
                return False
            
            if state is not None:
                job.state = state
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error
            if processing_time_ms is not None:
                job.processing_time_ms = processing_time_ms
            
            job.updated_at = datetime.utcnow()
            
            logger.info(f"Updated job {job_id} to state {job.state}")
            return True
    
    def cleanup_old_jobs(self) -> int:
        """
        Remove jobs older than cleanup threshold.
        
        Returns:
            Number of jobs removed
        """
        now = datetime.utcnow()
        cutoff = now - self._cleanup_after
        
        with self._lock:
            to_remove = [
                job_id for job_id, job in self._store.items()
                if job.updated_at < cutoff and job.state in [JobState.SUCCESS, JobState.FAILED]
            ]
            
            for job_id in to_remove:
                del self._store[job_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old jobs")
        
        return len(to_remove)
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about jobs in the store."""
        with self._lock:
            stats = {
                "total": len(self._store),
                "queued": sum(1 for j in self._store.values() if j.state == JobState.QUEUED),
                "processing": sum(1 for j in self._store.values() if j.state == JobState.PROCESSING),
                "success": sum(1 for j in self._store.values() if j.state == JobState.SUCCESS),
                "failed": sum(1 for j in self._store.values() if j.state == JobState.FAILED),
            }
        return stats


# Global job store instance
_job_store: Optional[JobStore] = None


def get_job_store(cleanup_after_hours: int = 24) -> JobStore:
    """
    Get or create the global job store instance.
    
    Args:
        cleanup_after_hours: Hours after which to cleanup old jobs
        
    Returns:
        JobStore instance
    """
    global _job_store
    if _job_store is None:
        _job_store = JobStore(cleanup_after_hours=cleanup_after_hours)
    return _job_store
