"""
FastAPI application for financial document classification.

Endpoints:
- GET  /health           - API health check
- POST /predict          - Classify text input
- POST /predict/document - Classify uploaded document (PDF/image)

Usage:
    uvicorn api.app:app --reload
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from src.confidence import (
    ClassificationResult,
    analyze_prediction,
    get_confidence_level,
    ConfidenceThresholds,
)
from src.ocr import extract_text, is_supported_file, get_supported_extensions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Model directory (relative to project root)
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "run1"

# Maximum file size for uploads (10 MB)
MAX_FILE_SIZE = 10 * 1024 * 1024


# ============================================================================
# Pydantic Models
# ============================================================================

class PredictRequest(BaseModel):
    """Request body for text prediction endpoint."""
    text: str = Field(
        ..., 
        min_length=1,
        description="The document text to classify",
        json_schema_extra={"example": "Invoice #INV-2024-001 for consulting services. Total: $5,250.00"}
    )


class TopPrediction(BaseModel):
    """Single prediction with probability."""
    label: str
    probability: float
    rank: int


class UncertaintyMetrics(BaseModel):
    """Uncertainty quantification metrics."""
    entropy: float = Field(..., description="Shannon entropy (higher = more uncertain)")
    margin: float = Field(..., description="Difference between top-1 and top-2 probabilities")


class PredictResponse(BaseModel):
    """Response body for prediction endpoints."""
    predicted_label: str = Field(..., description="Top predicted document type")
    confidence: float = Field(..., description="Confidence score (0-1)")
    confidence_level: str = Field(..., description="Human-readable confidence level")
    needs_review: bool = Field(..., description="Whether human review is recommended")
    review_reason: Optional[str] = Field(None, description="Reason for review recommendation")
    top_predictions: List[TopPrediction] = Field(..., description="Top-k predictions with probabilities")
    uncertainty_metrics: UncertaintyMetrics = Field(..., description="Uncertainty quantification metrics")


class DocumentPredictResponse(PredictResponse):
    """Response for document upload endpoint - includes extracted text info."""
    extracted_text_preview: str = Field(..., description="First 200 chars of extracted text")
    text_length: int = Field(..., description="Total length of extracted text")


class HealthResponse(BaseModel):
    """Response body for health check endpoint."""
    status: str
    model_loaded: bool
    model_path: str
    supported_file_types: List[str]
    confidence_thresholds: dict


# ============================================================================
# Model Artifacts
# ============================================================================

class ModelArtifacts:
    """Container for lazily loaded model artifacts."""
    
    def __init__(self):
        self.model: Optional[DistilBertForSequenceClassification] = None
        self.tokenizer: Optional[DistilBertTokenizerFast] = None
        self.label_classes: Optional[np.ndarray] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load(self) -> bool:
        """Load model artifacts from disk. Returns True if successful."""
        if not MODEL_DIR.exists():
            logger.error(f"Model directory not found: {MODEL_DIR}")
            return False
        
        try:
            if self.model is None:
                logger.info(f"Loading model from {MODEL_DIR}")
                self.model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
                self.model.to(self.device)
                self.model.eval()
            
            if self.tokenizer is None:
                self.tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
            
            if self.label_classes is None:
                classes_path = MODEL_DIR / "label_encoder.npy"
                if not classes_path.exists():
                    logger.error(f"Label encoder not found: {classes_path}")
                    return False
                self.label_classes = np.load(classes_path, allow_pickle=True)
            
            logger.info("Model artifacts loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if all artifacts are loaded."""
        return all([
            self.model is not None, 
            self.tokenizer is not None, 
            self.label_classes is not None
        ])


# Global artifacts instance
artifacts = ModelArtifacts()


# ============================================================================
# Application Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    logger.info("Starting Financial Document Classifier API...")
    if MODEL_DIR.exists():
        artifacts.load()
    else:
        logger.warning(f"Model not found at {MODEL_DIR}. Train a model first.")
    
    yield
    
    logger.info("Shutting down API...")


app = FastAPI(
    title="Financial Document Classifier",
    description="""
    AI-powered classification of financial documents using fine-tuned DistilBERT.
    
    ## Features
    - **Text Classification**: Classify document text directly
    - **Document Upload**: Upload PDFs or images for automatic text extraction and classification
    - **Uncertainty Quantification**: Get confidence scores and human review recommendations
    - **Multi-format Support**: PDF, PNG, JPG, TIFF, and plain text files
    
    ## Document Types
    - Invoice
    - Purchase Order
    - Bank Statement
    - Tax Notice
    - Contract
    - Other
    """,
    version="2.0.0",
    lifespan=lifespan
)


# ============================================================================
# Helper Functions
# ============================================================================

def ensure_model_loaded():
    """Ensure model is loaded, raise HTTP 503 if not available."""
    if not artifacts.is_loaded():
        if not artifacts.load():
            raise HTTPException(
                status_code=503,
                detail=f"Model not available. Please train the model first. Expected path: {MODEL_DIR}"
            )


def classify_text(text: str) -> ClassificationResult:
    """
    Classify text and return detailed results with uncertainty metrics.
    
    Args:
        text: Document text to classify
        
    Returns:
        ClassificationResult with predictions and confidence metrics
    """
    # Tokenize input
    encoding = artifacts.tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt",
    )
    
    # Move to device
    encoding = {k: v.to(artifacts.device) for k, v in encoding.items()}
    
    # Get prediction with uncertainty analysis
    with torch.no_grad():
        outputs = artifacts.model(**encoding)
        result = analyze_prediction(
            logits=outputs.logits,
            label_classes=artifacts.label_classes,
            top_k=3
        )
    
    return result


def result_to_response(result: ClassificationResult) -> dict:
    """Convert ClassificationResult to API response format."""
    return {
        "predicted_label": result.predicted_label,
        "confidence": round(result.confidence, 4),
        "confidence_level": get_confidence_level(result.confidence),
        "needs_review": result.needs_review,
        "review_reason": result.review_reason,
        "top_predictions": [
            {"label": p.label, "probability": round(p.probability, 4), "rank": p.rank}
            for p in result.top_predictions
        ],
        "uncertainty_metrics": {
            "entropy": round(result.entropy, 4),
            "margin": round(result.margin, 4)
        }
    }


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Check API health and model status.
    
    Returns information about model availability and supported file types.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=artifacts.is_loaded(),
        model_path=str(MODEL_DIR),
        supported_file_types=get_supported_extensions(),
        confidence_thresholds={
            "low_confidence": ConfidenceThresholds.LOW_CONFIDENCE,
            "high_confidence": ConfidenceThresholds.HIGH_CONFIDENCE,
            "min_margin": ConfidenceThresholds.MIN_MARGIN
        }
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Classify document text.
    
    Returns the predicted document type with confidence scores and
    uncertainty metrics. Flags predictions that may need human review.
    """
    ensure_model_loaded()
    
    try:
        result = classify_text(req.text)
        response_data = result_to_response(result)
        
        logger.info(
            f"Prediction: {result.predicted_label} "
            f"(confidence: {result.confidence:.1%}, needs_review: {result.needs_review})"
        )
        
        return PredictResponse(**response_data)
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/document", response_model=DocumentPredictResponse)
async def predict_document(file: UploadFile = File(..., description="Document file (PDF, PNG, JPG, etc.)")):
    """
    Classify an uploaded document file.
    
    Supports:
    - PDF files (with OCR fallback for scanned documents)
    - Image files (PNG, JPG, TIFF)
    - Plain text files
    
    The document is processed to extract text, then classified.
    Returns predictions with confidence scores and uncertainty metrics.
    """
    ensure_model_loaded()
    
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    if not is_supported_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {', '.join(get_supported_extensions())}"
        )
    
    # Read file content
    try:
        content = await file.read()
        
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)} MB"
            )
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
    
    # Extract text from document
    try:
        extracted_text = extract_text(content, file.filename)
        logger.info(f"Extracted {len(extracted_text)} characters from {file.filename}")
        
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ImportError as e:
        raise HTTPException(
            status_code=501,
            detail=f"OCR dependencies not installed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Text extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")
    
    # Classify extracted text
    try:
        result = classify_text(extracted_text)
        response_data = result_to_response(result)
        
        # Add document-specific fields
        response_data["extracted_text_preview"] = extracted_text[:200] + ("..." if len(extracted_text) > 200 else "")
        response_data["text_length"] = len(extracted_text)
        
        logger.info(
            f"Document {file.filename}: {result.predicted_label} "
            f"(confidence: {result.confidence:.1%}, needs_review: {result.needs_review})"
        )
        
        return DocumentPredictResponse(**response_data)
    
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")
