"""
FastAPI application for financial document classification.

Usage:
    uvicorn api.app:app --reload
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Model directory (relative to project root)
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "run1"


class PredictRequest(BaseModel):
    """Request body for prediction endpoint."""
    text: str = Field(
        ..., 
        min_length=1,
        description="The document text to classify",
        json_schema_extra={"example": "Invoice #INV-2024-001 for consulting services. Total: $5,250.00"}
    )


class PredictResponse(BaseModel):
    """Response body for prediction endpoint."""
    label: str = Field(..., description="Predicted document type")
    confidence: float = Field(..., description="Confidence score (0-1)")


class HealthResponse(BaseModel):
    """Response body for health check endpoint."""
    status: str
    model_loaded: bool
    model_path: str


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
    description="API for classifying financial documents using a fine-tuned DistilBERT model.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        model_loaded=artifacts.is_loaded(),
        model_path=str(MODEL_DIR)
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Classify a financial document.
    
    Returns the predicted document type and confidence score.
    """
    # Ensure model is loaded
    if not artifacts.is_loaded():
        if not artifacts.load():
            raise HTTPException(
                status_code=503,
                detail=f"Model not available. Please train the model first. Expected path: {MODEL_DIR}"
            )
    
    try:
        # Tokenize input
        encoding = artifacts.tokenizer(
            req.text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        )
        
        # Move to device
        encoding = {k: v.to(artifacts.device) for k, v in encoding.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = artifacts.model(**encoding)
            logits = outputs.logits.cpu()
            
            # Calculate softmax probabilities
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
            
            label = str(artifacts.label_classes[pred_idx.item()])
            conf_score = round(confidence.item(), 4)
        
        logger.info(f"Prediction: {label} (confidence: {conf_score})")
        
        return PredictResponse(label=label, confidence=conf_score)
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
