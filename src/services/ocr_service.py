"""
OCR service for text extraction from documents.

Wrapper around the existing OCR module.
"""

from src.ocr import extract_text as _extract_text
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class OCRError(Exception):
    """Base exception for OCR errors."""
    pass


def extract_text(file_content: bytes, filename: str) -> str:
    """
    Extract text from document file.
    
    Args:
        file_content: Raw file bytes
        filename: Original filename
        
    Returns:
        Extracted text
        
    Raises:
        OCRError: If extraction fails
    """
    try:
        logger.info(f"Extracting text from {filename}")
        text = _extract_text(file_content, filename)
        logger.info(f"Extracted {len(text)} characters from {filename}")
        return text
    except Exception as e:
        logger.error(f"Text extraction failed for {filename}: {e}")
        raise OCRError(f"Failed to extract text: {str(e)}")
