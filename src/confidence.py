"""
Confidence scoring and uncertainty quantification module.

Provides:
- Calibrated confidence scores
- Human review flagging for uncertain predictions
- Top-k predictions with probabilities
- Prediction quality assessment
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# Confidence thresholds for decision making
class ConfidenceThresholds:
    """Configurable thresholds for prediction confidence."""
    
    # Below this: definitely needs human review
    LOW_CONFIDENCE = 0.5
    
    # Above this: high confidence, no review needed
    HIGH_CONFIDENCE = 0.85
    
    # Margin between top-2 predictions should be at least this
    # (if top prediction is 60% and second is 55%, margin is too low)
    MIN_MARGIN = 0.15


@dataclass
class PredictionDetail:
    """Detailed information about a single class prediction."""
    label: str
    probability: float
    rank: int


@dataclass
class ClassificationResult:
    """
    Complete classification result with uncertainty quantification.
    
    Attributes:
        predicted_label: The top predicted class label
        confidence: Probability of the predicted class (0-1)
        needs_review: Whether human review is recommended
        review_reason: Explanation for why review is needed (if applicable)
        top_predictions: List of top-k predictions with probabilities
        entropy: Entropy of the prediction distribution (higher = more uncertain)
        margin: Difference between top-1 and top-2 probabilities
    """
    predicted_label: str
    confidence: float
    needs_review: bool
    review_reason: Optional[str]
    top_predictions: List[PredictionDetail]
    entropy: float
    margin: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "predicted_label": self.predicted_label,
            "confidence": round(self.confidence, 4),
            "needs_review": self.needs_review,
            "review_reason": self.review_reason,
            "top_predictions": [
                {"label": p.label, "probability": round(p.probability, 4), "rank": p.rank}
                for p in self.top_predictions
            ],
            "uncertainty_metrics": {
                "entropy": round(self.entropy, 4),
                "margin": round(self.margin, 4)
            }
        }


def compute_entropy(probabilities: np.ndarray) -> float:
    """
    Compute Shannon entropy of a probability distribution.
    
    Higher entropy = more uncertainty (predictions spread across classes).
    Lower entropy = more certainty (one class dominates).
    
    Args:
        probabilities: Array of class probabilities (should sum to 1)
        
    Returns:
        Entropy value (0 = perfectly certain, log(n_classes) = maximum uncertainty)
    """
    # Avoid log(0) by clipping
    probs = np.clip(probabilities, 1e-10, 1.0)
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy)


def compute_margin(probabilities: np.ndarray) -> float:
    """
    Compute margin between top-1 and top-2 predictions.
    
    Higher margin = more confident in the top prediction.
    Lower margin = confusion between top classes.
    
    Args:
        probabilities: Array of class probabilities
        
    Returns:
        Margin value (0-1)
    """
    if len(probabilities) < 2:
        return 1.0
    
    sorted_probs = np.sort(probabilities)[::-1]
    margin = sorted_probs[0] - sorted_probs[1]
    return float(margin)


def should_flag_for_review(
    confidence: float,
    margin: float,
    entropy: float,
    n_classes: int
) -> tuple[bool, Optional[str]]:
    """
    Determine if a prediction should be flagged for human review.
    
    Uses multiple signals:
    1. Low confidence in top prediction
    2. Small margin between top predictions (model is confused)
    3. High entropy (uncertainty spread across many classes)
    
    Args:
        confidence: Probability of top predicted class
        margin: Difference between top-1 and top-2 probabilities
        entropy: Entropy of the prediction distribution
        n_classes: Total number of classes
        
    Returns:
        Tuple of (needs_review: bool, reason: Optional[str])
    """
    # Normalize entropy to 0-1 scale (max entropy = log(n_classes))
    max_entropy = np.log(n_classes)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    # Check various conditions for flagging
    if confidence < ConfidenceThresholds.LOW_CONFIDENCE:
        return True, f"Low confidence ({confidence:.1%})"
    
    if margin < ConfidenceThresholds.MIN_MARGIN:
        return True, f"Ambiguous prediction (margin: {margin:.1%})"
    
    if normalized_entropy > 0.7:  # High uncertainty
        return True, f"High uncertainty (entropy: {normalized_entropy:.2f})"
    
    # Borderline confidence with additional concerns
    if confidence < ConfidenceThresholds.HIGH_CONFIDENCE:
        if margin < 0.25 or normalized_entropy > 0.5:
            return True, f"Borderline confidence ({confidence:.1%}) with ambiguity"
    
    return False, None


def analyze_prediction(
    logits: torch.Tensor,
    label_classes: np.ndarray,
    top_k: int = 3
) -> ClassificationResult:
    """
    Analyze model output with full uncertainty quantification.
    
    Args:
        logits: Raw model output logits (shape: [1, n_classes])
        label_classes: Array of class labels
        top_k: Number of top predictions to include
        
    Returns:
        ClassificationResult with confidence metrics and review recommendation
    """
    # Convert logits to probabilities
    with torch.no_grad():
        probabilities = F.softmax(logits, dim=1).cpu().numpy().flatten()
    
    n_classes = len(label_classes)
    top_k = min(top_k, n_classes)
    
    # Get top-k predictions
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    
    top_predictions = [
        PredictionDetail(
            label=str(label_classes[idx]),
            probability=float(probabilities[idx]),
            rank=rank + 1
        )
        for rank, idx in enumerate(top_indices)
    ]
    
    # Extract key metrics
    predicted_label = top_predictions[0].label
    confidence = top_predictions[0].probability
    
    entropy = compute_entropy(probabilities)
    margin = compute_margin(probabilities)
    
    # Determine if review is needed
    needs_review, review_reason = should_flag_for_review(
        confidence=confidence,
        margin=margin,
        entropy=entropy,
        n_classes=n_classes
    )
    
    result = ClassificationResult(
        predicted_label=predicted_label,
        confidence=confidence,
        needs_review=needs_review,
        review_reason=review_reason,
        top_predictions=top_predictions,
        entropy=entropy,
        margin=margin
    )
    
    if needs_review:
        logger.info(f"Prediction flagged for review: {review_reason}")
    else:
        logger.debug(f"High-confidence prediction: {predicted_label} ({confidence:.1%})")
    
    return result


def get_confidence_level(confidence: float) -> str:
    """
    Get human-readable confidence level label.
    
    Args:
        confidence: Probability value (0-1)
        
    Returns:
        One of: "very_high", "high", "medium", "low", "very_low"
    """
    if confidence >= 0.95:
        return "very_high"
    elif confidence >= 0.85:
        return "high"
    elif confidence >= 0.70:
        return "medium"
    elif confidence >= 0.50:
        return "low"
    else:
        return "very_low"

