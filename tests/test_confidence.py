"""
Tests for the confidence and uncertainty quantification module.

Run with: pytest tests/test_confidence.py -v
"""

import numpy as np
import pytest
import torch

from src.confidence import (
    compute_entropy,
    compute_margin,
    should_flag_for_review,
    analyze_prediction,
    get_confidence_level,
    ConfidenceThresholds,
    ClassificationResult,
)


class TestComputeEntropy:
    """Tests for entropy calculation."""
    
    def test_entropy_certain_prediction(self):
        """Entropy should be 0 for a perfectly certain prediction."""
        # All probability on one class
        probs = np.array([1.0, 0.0, 0.0, 0.0])
        entropy = compute_entropy(probs)
        assert entropy < 0.01  # Near zero
    
    def test_entropy_uniform_distribution(self):
        """Entropy should be maximum for uniform distribution."""
        # Equal probability across all classes
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        entropy = compute_entropy(probs)
        max_entropy = np.log(4)  # log(n_classes)
        assert abs(entropy - max_entropy) < 0.01
    
    def test_entropy_partial_uncertainty(self):
        """Entropy should be between 0 and max for partial uncertainty."""
        probs = np.array([0.7, 0.2, 0.1])
        entropy = compute_entropy(probs)
        assert 0 < entropy < np.log(3)


class TestComputeMargin:
    """Tests for margin calculation."""
    
    def test_margin_high_confidence(self):
        """Margin should be high when top prediction dominates."""
        probs = np.array([0.95, 0.03, 0.02])
        margin = compute_margin(probs)
        assert margin == pytest.approx(0.92, abs=0.01)
    
    def test_margin_confused_prediction(self):
        """Margin should be low when top predictions are close."""
        probs = np.array([0.35, 0.33, 0.32])
        margin = compute_margin(probs)
        assert margin == pytest.approx(0.02, abs=0.01)
    
    def test_margin_single_class(self):
        """Margin should be 1.0 for single class."""
        probs = np.array([1.0])
        margin = compute_margin(probs)
        assert margin == 1.0


class TestShouldFlagForReview:
    """Tests for human review flagging logic."""
    
    def test_low_confidence_flagged(self):
        """Low confidence predictions should be flagged."""
        needs_review, reason = should_flag_for_review(
            confidence=0.4,
            margin=0.3,
            entropy=0.5,
            n_classes=6
        )
        assert needs_review is True
        assert "Low confidence" in reason
    
    def test_high_confidence_not_flagged(self):
        """High confidence predictions should not be flagged."""
        needs_review, reason = should_flag_for_review(
            confidence=0.95,
            margin=0.8,
            entropy=0.2,
            n_classes=6
        )
        assert needs_review is False
        assert reason is None
    
    def test_low_margin_flagged(self):
        """Predictions with low margin should be flagged."""
        needs_review, reason = should_flag_for_review(
            confidence=0.6,
            margin=0.05,  # Very low margin
            entropy=0.5,
            n_classes=6
        )
        assert needs_review is True
        assert "Ambiguous" in reason or "margin" in reason.lower()
    
    def test_high_entropy_flagged(self):
        """Predictions with high entropy should be flagged."""
        max_entropy = np.log(6)
        high_entropy = max_entropy * 0.8  # 80% of max
        
        needs_review, reason = should_flag_for_review(
            confidence=0.6,
            margin=0.2,
            entropy=high_entropy,
            n_classes=6
        )
        assert needs_review is True


class TestAnalyzePrediction:
    """Tests for the full prediction analysis."""
    
    def test_analyze_returns_classification_result(self):
        """analyze_prediction should return ClassificationResult."""
        # Create mock logits (6 classes)
        logits = torch.tensor([[2.0, 0.5, 0.3, 0.1, 0.1, 0.1]])
        labels = np.array(['invoice', 'contract', 'bank_statement', 'other', 'purchase_order', 'tax_notice'])
        
        result = analyze_prediction(logits, labels, top_k=3)
        
        assert isinstance(result, ClassificationResult)
        assert result.predicted_label == 'invoice'  # Highest logit
        assert 0 <= result.confidence <= 1
        assert len(result.top_predictions) == 3
    
    def test_analyze_top_k_ordering(self):
        """Top-k predictions should be ordered by probability."""
        logits = torch.tensor([[1.0, 3.0, 2.0, 0.5, 0.5, 0.5]])
        labels = np.array(['a', 'b', 'c', 'd', 'e', 'f'])
        
        result = analyze_prediction(logits, labels, top_k=3)
        
        # Should be ordered: b (3.0), c (2.0), a (1.0)
        assert result.top_predictions[0].label == 'b'
        assert result.top_predictions[1].label == 'c'
        assert result.top_predictions[2].label == 'a'
        
        # Probabilities should be decreasing
        probs = [p.probability for p in result.top_predictions]
        assert probs == sorted(probs, reverse=True)
    
    def test_analyze_includes_metrics(self):
        """Result should include entropy and margin metrics."""
        logits = torch.tensor([[2.0, 1.0, 0.5, 0.3, 0.2, 0.1]])
        labels = np.array(['a', 'b', 'c', 'd', 'e', 'f'])
        
        result = analyze_prediction(logits, labels)
        
        assert result.entropy >= 0
        assert 0 <= result.margin <= 1


class TestGetConfidenceLevel:
    """Tests for confidence level labels."""
    
    def test_very_high_confidence(self):
        assert get_confidence_level(0.98) == "very_high"
    
    def test_high_confidence(self):
        assert get_confidence_level(0.88) == "high"
    
    def test_medium_confidence(self):
        assert get_confidence_level(0.75) == "medium"
    
    def test_low_confidence(self):
        assert get_confidence_level(0.55) == "low"
    
    def test_very_low_confidence(self):
        assert get_confidence_level(0.30) == "very_low"


class TestConfidenceThresholds:
    """Tests for threshold configuration."""
    
    def test_thresholds_are_valid(self):
        """Thresholds should be in valid range."""
        assert 0 < ConfidenceThresholds.LOW_CONFIDENCE < 1
        assert 0 < ConfidenceThresholds.HIGH_CONFIDENCE < 1
        assert 0 < ConfidenceThresholds.MIN_MARGIN < 1
    
    def test_low_less_than_high(self):
        """Low threshold should be less than high threshold."""
        assert ConfidenceThresholds.LOW_CONFIDENCE < ConfidenceThresholds.HIGH_CONFIDENCE


class TestClassificationResultToDict:
    """Tests for result serialization."""
    
    def test_to_dict_structure(self):
        """to_dict should return proper structure."""
        from src.confidence import PredictionDetail
        
        result = ClassificationResult(
            predicted_label="invoice",
            confidence=0.92,
            needs_review=False,
            review_reason=None,
            top_predictions=[
                PredictionDetail(label="invoice", probability=0.92, rank=1),
                PredictionDetail(label="contract", probability=0.05, rank=2),
            ],
            entropy=0.35,
            margin=0.87
        )
        
        d = result.to_dict()
        
        assert d["predicted_label"] == "invoice"
        assert d["confidence"] == 0.92
        assert d["needs_review"] is False
        assert len(d["top_predictions"]) == 2
        assert "uncertainty_metrics" in d
        assert d["uncertainty_metrics"]["entropy"] == 0.35

