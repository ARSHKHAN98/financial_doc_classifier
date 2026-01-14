"""
API authentication and rate limiting middleware.
"""

import time
from typing import Dict, Optional
from collections import defaultdict
from threading import Lock
from fastapi import Security, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from src.config.settings import settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# API Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class RateLimiter:
    """
    In-memory rate limiter.
    
    Designed to be easily replaceable with Redis for production.
    """
    
    def __init__(self, max_requests: int, window_seconds: int):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests allowed per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, list] = defaultdict(list)
        self._lock = Lock()
    
    def is_allowed(self, key: str) -> tuple[bool, Optional[int]]:
        """
        Check if request is allowed for the given key.
        
        Args:
            key: Rate limit key (e.g., API key)
            
        Returns:
            Tuple of (is_allowed, retry_after_seconds)
        """
        now = time.time()
        window_start = now - self.window_seconds
        
        with self._lock:
            # Remove old requests outside the window
            self._requests[key] = [
                req_time for req_time in self._requests[key]
                if req_time > window_start
            ]
            
            # Check if under limit
            if len(self._requests[key]) >= self.max_requests:
                # Calculate retry-after
                oldest_request = self._requests[key][0]
                retry_after = int(oldest_request + self.window_seconds - now) + 1
                return False, retry_after
            
            # Add current request
            self._requests[key].append(now)
            return True, None
    
    def get_usage(self, key: str) -> Dict[str, int]:
        """Get current usage stats for a key."""
        now = time.time()
        window_start = now - self.window_seconds
        
        with self._lock:
            # Clean old requests
            self._requests[key] = [
                req_time for req_time in self._requests[key]
                if req_time > window_start
            ]
            
            return {
                "current": len(self._requests[key]),
                "limit": self.max_requests,
                "window_seconds": self.window_seconds
            }


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(
            max_requests=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window
        )
    return _rate_limiter


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Verify API key from header.
    
    Args:
        api_key: API key from X-API-Key header
        
    Returns:
        Valid API key
        
    Raises:
        HTTPException: If API key is missing or invalid
    """
    if not api_key:
        logger.warning("Request missing API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header."
        )
    
    # Get valid API keys from settings
    valid_keys = settings.get_api_keys_set()
    
    if api_key not in valid_keys:
        logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return api_key


async def check_rate_limit(request: Request, api_key: str) -> None:
    """
    Check rate limit for the API key.
    
    Args:
        request: FastAPI request
        api_key: Validated API key
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    rate_limiter = get_rate_limiter()
    allowed, retry_after = rate_limiter.is_allowed(api_key)
    
    if not allowed:
        logger.warning(f"Rate limit exceeded for API key: {api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)}
        )
    
    # Add rate limit info to request state
    usage = rate_limiter.get_usage(api_key)
    request.state.rate_limit_usage = usage
