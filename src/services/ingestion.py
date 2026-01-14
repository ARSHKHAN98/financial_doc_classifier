"""
Document ingestion service.

Handles file uploads, validation, and text extraction.
"""

from typing import Tuple
from pathlib import Path

from src.config.settings import settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class IngestionError(Exception):
    """Base exception for ingestion errors."""
    pass


class FileTooLargeError(IngestionError):
    """Raised when file exceeds size limit."""
    pass


class UnsupportedFileTypeError(IngestionError):
    """Raised when file type is not supported."""
    pass


class EmptyFileError(IngestionError):
    """Raised when file is empty."""
    pass


def validate_file(filename: str, content: bytes) -> None:
    """
    Validate uploaded file.
    
    Args:
        filename: Original filename
        content: File content bytes
        
    Raises:
        EmptyFileError: If file is empty
        FileTooLargeError: If file exceeds size limit
        UnsupportedFileTypeError: If file type not supported
    """
    # Check if empty
    if len(content) == 0:
        raise EmptyFileError("File is empty")
    
    # Check size
    if len(content) > settings.max_file_size_bytes:
        raise FileTooLargeError(
            f"File size ({len(content)} bytes) exceeds maximum "
            f"({settings.max_file_size_bytes} bytes)"
        )
    
    # Check file type
    suffix = Path(filename).suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise UnsupportedFileTypeError(
            f"File type '{suffix}' not supported. "
            f"Allowed types: {', '.join(settings.allowed_extensions)}"
        )


def is_supported_file(filename: str) -> bool:
    """
    Check if file type is supported.
    
    Args:
        filename: Filename to check
        
    Returns:
        True if supported, False otherwise
    """
    suffix = Path(filename).suffix.lower()
    return suffix in settings.allowed_extensions


def get_supported_extensions() -> list:
    """Get list of supported file extensions."""
    return list(settings.allowed_extensions)
