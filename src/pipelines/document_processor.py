"""
Document processing pipeline.

Orchestrates the full document processing workflow.
"""

import time
import asyncio
from typing import Dict, Any, Optional

from src.services import (
    validate_file,
    extract_text,
    classify_text,
    IngestionError,
    OCRError,
    ClassificationError,
)
from src.pipelines.job_manager import get_job_store, JobState
from src.config.settings import settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


async def process_document_async(
    job_id: str,
    file_content: bytes,
    filename: str
) -> None:
    """
    Process a document asynchronously.
    
    This is the main processing pipeline that:
    1. Validates the file
    2. Extracts text (OCR if needed)
    3. Classifies the document
    4. Updates job state
    
    Args:
        job_id: Job ID
        file_content: Raw file bytes
        filename: Original filename
    """
    job_store = get_job_store()
    start_time = time.time()
    
    try:
        # Update to processing state
        job_store.update_job(job_id, state=JobState.PROCESSING)
        logger.info(f"Job {job_id}: Starting processing for {filename}")
        
        # Step 1: Validate file
        try:
            validate_file(filename, file_content)
        except IngestionError as e:
            raise ValueError(f"Validation failed: {str(e)}")
        
        # Step 2: Extract text (run in thread pool for CPU-bound OCR)
        try:
            text = await asyncio.to_thread(extract_text, file_content, filename)
            
            if not text or len(text.strip()) < 10:
                raise ValueError("Extracted text is too short or empty")
                
        except OCRError as e:
            raise ValueError(f"Text extraction failed: {str(e)}")
        
        # Step 3: Classify text
        try:
            result = await asyncio.to_thread(classify_text, text)
            
            # Add extracted text info to result
            result["extracted_text_length"] = len(text)
            result["extracted_text_preview"] = text[:200] + ("..." if len(text) > 200 else "")
            
        except ClassificationError as e:
            raise ValueError(f"Classification failed: {str(e)}")
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Update job with success
        job_store.update_job(
            job_id,
            state=JobState.SUCCESS,
            result=result,
            processing_time_ms=processing_time_ms
        )
        
        logger.info(
            f"Job {job_id}: Completed successfully in {processing_time_ms}ms. "
            f"Result: {result.get('predicted_label')}"
        )
        
    except asyncio.TimeoutError:
        error_msg = f"Processing timeout after {settings.job_timeout_seconds}s"
        logger.error(f"Job {job_id}: {error_msg}")
        job_store.update_job(job_id, state=JobState.FAILED, error=error_msg)
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Job {job_id}: Processing failed - {error_msg}", exc_info=True)
        job_store.update_job(job_id, state=JobState.FAILED, error=error_msg)


def process_text_sync(text: str) -> Dict[str, Any]:
    """
    Process text synchronously (for text-only endpoint).
    
    Args:
        text: Document text
        
    Returns:
        Classification result
        
    Raises:
        ClassificationError: If classification fails
    """
    return classify_text(text)
