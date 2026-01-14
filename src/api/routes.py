"""
API routes for document processing.
"""

import asyncio
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks, Request
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from src.api.schemas import (
    TextClassifyRequest,
    TextClassifyResponse,
    DocumentUploadResponse,
    JobStatusResponse,
    HealthResponse,
    ReadyResponse,
)
from src.api.auth import verify_api_key, check_rate_limit
from src.api.metrics import jobs_total, predictions_total
from src.pipelines import (
    get_job_store,
    process_document_async,
    process_text_sync,
    JobState,
)
from src.models.inference import get_model
from src.config.settings import settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1")


# ============================================================================
# Health & Readiness Endpoints (No Auth Required)
# ============================================================================

@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns basic service status and configuration.
    No authentication required.
    """
    model = get_model(settings.model_dir)
    
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        model_loaded=model.is_loaded()
    )


@router.get("/ready", response_model=ReadyResponse, tags=["Health"])
async def readiness_check():
    """
    Readiness check endpoint.
    
    Returns whether the service is ready to handle requests.
    No authentication required.
    """
    model = get_model(settings.model_dir)
    model_ready = model.is_loaded()
    
    return ReadyResponse(
        ready=model_ready,
        checks={
            "model_loaded": model_ready,
        }
    )


# ============================================================================
# Text Classification Endpoint
# ============================================================================

@router.post("/classify/text", response_model=TextClassifyResponse, tags=["Classification"])
async def classify_text(
    request: Request,
    req: TextClassifyRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Classify document text (synchronous).
    
    Returns immediate classification results.
    Requires API key authentication.
    """
    # Check rate limit
    await check_rate_limit(request, api_key)
    
    try:
        # Process text synchronously
        result = process_text_sync(req.text)
        
        # Update metrics
        predictions_total.labels(
            predicted_label=result["predicted_label"],
            needs_review=str(result["needs_review"])
        ).inc()
        
        logger.info(f"Text classified: {result['predicted_label']}")
        
        return TextClassifyResponse(**result)
        
    except Exception as e:
        logger.error(f"Text classification failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


# ============================================================================
# Document Upload & Job Management Endpoints
# ============================================================================

@router.post("/documents", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file to process"),
    api_key: str = Depends(verify_api_key)
):
    """
    Upload a document for asynchronous processing.
    
    Returns a job ID that can be used to track processing status.
    Requires API key authentication.
    """
    # Check rate limit
    await check_rate_limit(request, api_key)
    
    # Validate filename
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    # Read file content
    try:
        content = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(status_code=400, detail="Failed to read file")
    
    # Create job
    job_store = get_job_store(settings.job_cleanup_after_hours)
    job_id = job_store.create_job(filename=file.filename)
    
    # Update metrics
    jobs_total.labels(state=JobState.QUEUED.value).inc()
    
    # Schedule background processing with timeout
    async def process_with_timeout():
        try:
            await asyncio.wait_for(
                process_document_async(job_id, content, file.filename),
                timeout=settings.job_timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error(f"Job {job_id} timed out after {settings.job_timeout_seconds}s")
            job_store.update_job(
                job_id,
                state=JobState.FAILED,
                error=f"Processing timeout after {settings.job_timeout_seconds}s"
            )
    
    background_tasks.add_task(process_with_timeout)
    
    logger.info(f"Document upload created job: {job_id}")
    
    return DocumentUploadResponse(
        job_id=job_id,
        status="queued",
        message=f"Document queued for processing. Use GET /v1/documents/{job_id} to check status."
    )


@router.get("/documents/{job_id}", response_model=JobStatusResponse, tags=["Documents"])
async def get_job_status(
    request: Request,
    job_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Get job status and result.
    
    Returns the current state of the job and results if completed.
    Requires API key authentication.
    """
    # Check rate limit
    await check_rate_limit(request, api_key)
    
    job_store = get_job_store()
    job = job_store.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return JobStatusResponse(**job.to_dict())


# ============================================================================
# Metrics Endpoint (No Auth for Prometheus scraping)
# ============================================================================

@router.get("/metrics", tags=["Observability"])
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus format.
    No authentication required for monitoring systems.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
