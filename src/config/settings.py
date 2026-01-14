"""
Application configuration management.

Loads settings from environment variables with sensible defaults.
"""

import os
from typing import List
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = "Document Processing Service"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # Security
    api_keys: List[str] = ["dev-key-12345"]  # Comma-separated in env
    cors_origins: List[str] = ["*"]
    
    # Rate Limiting
    rate_limit_requests: int = 60  # requests per window
    rate_limit_window: int = 60    # window size in seconds
    
    # File Upload
    max_file_size_mb: int = 10
    allowed_extensions: List[str] = [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".txt"]
    
    # ML Model
    model_dir: Path = Path(__file__).resolve().parent.parent.parent / "models" / "run1"
    model_timeout_seconds: int = 30
    
    # Job Processing
    job_timeout_seconds: int = 120
    job_cleanup_after_hours: int = 24
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # or "text"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Allow parsing of comma-separated lists
        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str):
            if field_name in ["api_keys", "cors_origins", "allowed_extensions"]:
                return [x.strip() for x in raw_val.split(",")]
            return raw_val
    
    @property
    def max_file_size_bytes(self) -> int:
        """Convert MB to bytes."""
        return self.max_file_size_mb * 1024 * 1024
    
    def get_api_keys_set(self) -> set:
        """Return API keys as a set for O(1) lookup."""
        # Load from env var if set, otherwise use default
        env_keys = os.getenv("API_KEYS", "")
        if env_keys:
            keys = [k.strip() for k in env_keys.split(",") if k.strip()]
            return set(keys)
        return set(self.api_keys)


# Global settings instance
settings = Settings()
