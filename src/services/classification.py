"""
Document classification service.

Handles text classification using the ML model.
"""

from typing import Dict, Any
from src.models.inference import get_model
from src.confidence import analyze_prediction
from src.config.settings import settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ClassificationError(Exception):
    """Base exception for classification errors."""
    pass


def classify_text(text: str) -> Dict[str, Any]:
    """
    Classify document text.
    
    Args:
        text: Document text to classify
        
    Returns:
        Dictionary with classification results including:
        - predicted_label: str
        - confidence: float
        - confidence_level: str
        - needs_review: bool
        - review_reason: str or None
        - top_predictions: list
        - uncertainty_metrics: dict
        
    Raises:
        ClassificationError: If classification fails
    """
    try:
        # Get model instance
        model = get_model(settings.model_dir)
        
        if not model.is_loaded():
            raise ClassificationError("Model not loaded")
        
        # Run inference
        logits = model.predict(text)
        
        # Analyze prediction with uncertainty quantification
        result = analyze_prediction(
            logits=logits,
            label_classes=model.get_label_classes(),
            top_k=3
        )
        
        # Convert to dictionary
        return result.to_dict()
        
    except ClassificationError:
        raise
    except Exception as e:
        logger.error(f"Classification failed: {e}", exc_info=True)
        raise ClassificationError(f"Classification failed: {str(e)}")
