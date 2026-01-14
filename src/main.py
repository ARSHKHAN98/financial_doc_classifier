"""
Main FastAPI application.

Production-grade document processing backend service.
"""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from src.api import router
from src.api.metrics import http_requests_total, http_request_duration_seconds
from src.config.settings import settings
from src.models.inference import get_model
from src.utils.logging_config import setup_logging, get_logger, request_id_context
from src.pipelines import get_job_store

# Setup logging
setup_logging(log_level=settings.log_level, log_format=settings.log_format)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Model directory: {settings.model_dir}")
    
    # Load ML model
    model = get_model(settings.model_dir)
    if settings.model_dir.exists():
        success = model.load()
        if success:
            logger.info("ML model loaded successfully")
        else:
            logger.warning("ML model failed to load - service will return 503 for predictions")
    else:
        logger.warning(f"Model directory not found: {settings.model_dir}")
    
    # Initialize job store
    job_store = get_job_store(settings.job_cleanup_after_hours)
    logger.info("Job store initialized")
    
    logger.info(f"Service ready on {settings.host}:{settings.port}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down service...")
    logger.info("Cleanup complete")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title=settings.app_name,
        description="""
        Production-grade Document Processing Backend Service.
        
        ## Features
        - **Async Job Processing**: Upload documents and track processing via job IDs
        - **Text Classification**: Direct text classification endpoint
        - **Multi-format Support**: PDF, images (PNG, JPG, TIFF), and text files
        - **OCR Support**: Automatic text extraction from scanned documents
        - **Authentication**: API key-based authentication via X-API-Key header
        - **Rate Limiting**: Per-key rate limiting to prevent abuse
        - **Observability**: Health checks, readiness probes, and Prometheus metrics
        - **Uncertainty Quantification**: Confidence scores and human review flags
        
        ## Authentication
        All endpoints (except /health, /ready, /metrics) require an API key.
        Include your API key in the `X-API-Key` header.
        
        ## Rate Limits
        Default: 60 requests per minute per API key.
        
        ## Document Types Supported
        - Invoice
        - Purchase Order
        - Bank Statement
        - Tax Notice
        - Contract
        - Other
        """,
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Request ID and logging middleware
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next):
        """Add request ID and logging for all requests."""
        # Generate request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request_id_context.set(request_id)
        
        # Start timer
        start_time = time.time()
        
        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(f"Request failed: {e}", exc_info=True)
            response = JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error"}
            )
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        # Log request
        logger.info(
            f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s"
        )
        
        # Update metrics
        http_requests_total.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        http_request_duration_seconds.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        return response
    
    # Exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors."""
        logger.warning(f"Validation error: {exc}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exc.errors()}
        )
    
    # Include API routes
    app.include_router(router)
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API information."""
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "status": "operational",
            "documentation": "/docs",
            "health": "/v1/health",
            "metrics": "/v1/metrics"
        }
    
    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
