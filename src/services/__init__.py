"""Services module."""

from .ingestion import (
    validate_file,
    is_supported_file,
    get_supported_extensions,
    IngestionError,
    FileTooLargeError,
    UnsupportedFileTypeError,
    EmptyFileError,
)
from .ocr_service import extract_text, OCRError
from .classification import classify_text, ClassificationError

__all__ = [
    "validate_file",
    "is_supported_file",
    "get_supported_extensions",
    "IngestionError",
    "FileTooLargeError",
    "UnsupportedFileTypeError",
    "EmptyFileError",
    "extract_text",
    "OCRError",
    "classify_text",
    "ClassificationError",
]
