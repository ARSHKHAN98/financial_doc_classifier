"""
API request/response schemas using Pydantic.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from src.pipelines.job_manager import JobState


# ============================================================================
# Request Schemas
# ============================================================================

class TextClassifyRequest(BaseModel):
    """Request for text classification."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Document text to classify",
        json_schema_extra={"example": "Invoice #INV-2024-001 for consulting services. Total: $5,250.00"}
    )


# ============================================================================
# Response Schemas
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")


class ReadyResponse(BaseModel):
    """Readiness check response."""
    ready: bool = Field(..., description="Whether service is ready to handle requests")
    checks: Dict[str, bool] = Field(..., description="Individual readiness checks")


class TopPrediction(BaseModel):
    """Individual prediction with probability."""
    label: str
    probability: float
    rank: int


class UncertaintyMetrics(BaseModel):
    """Uncertainty quantification metrics."""
    entropy: float = Field(..., description="Shannon entropy (higher = more uncertain)")
    margin: float = Field(..., description="Difference between top-1 and top-2 probabilities")


class ClassificationResult(BaseModel):
    """Classification result."""
    predicted_label: str = Field(..., description="Predicted document type")
    confidence: float = Field(..., description="Confidence score (0-1)")
    confidence_level: str = Field(..., description="Human-readable confidence level")
    needs_review: bool = Field(..., description="Whether human review is recommended")
    review_reason: Optional[str] = Field(None, description="Reason for review recommendation")
    top_predictions: List[TopPrediction] = Field(..., description="Top-k predictions")
    uncertainty_metrics: UncertaintyMetrics = Field(..., description="Uncertainty metrics")


class TextClassifyResponse(ClassificationResult):
    """Response for text classification endpoint."""
    pass


class DocumentUploadResponse(BaseModel):
    """Response for document upload endpoint."""
    job_id: str = Field(..., description="Job ID for tracking")
    status: str = Field(..., description="Initial job status")
    message: str = Field(..., description="Status message")


class JobStatusResponse(BaseModel):
    """Response for job status endpoint."""
    job_id: str
    state: str
    created_at: str
    updated_at: str
    filename: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: Optional[int] = None


class ErrorResponse(BaseModel):
    """Error response."""
    detail: str = Field(..., description="Error message")
    error_type: Optional[str] = Field(None, description="Error type")
