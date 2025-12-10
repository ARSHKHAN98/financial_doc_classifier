"""
Tests for the FastAPI prediction endpoint.

Run with: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient

from api.app import app, artifacts, MODEL_DIR


client = TestClient(app)


class TestHealthEndpoint:
    """Tests for the /health endpoint."""
    
    def test_health_returns_200(self):
        """Health endpoint should always return 200."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_response_structure(self):
        """Health response should have expected fields."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "model_path" in data
        assert data["status"] == "healthy"


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""
    
    def test_predict_rejects_empty_text(self):
        """Predict should reject empty text input."""
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422  # Validation error
    
    def test_predict_rejects_missing_text(self):
        """Predict should reject requests without text field."""
        response = client.post("/predict", json={})
        assert response.status_code == 422
    
    @pytest.mark.skipif(
        not MODEL_DIR.exists(),
        reason="Model not trained yet - run training first"
    )
    def test_predict_returns_label(self):
        """Predict should return a label and confidence for valid input."""
        response = client.post(
            "/predict", 
            json={"text": "Invoice #12345 for consulting services. Amount due: $1,000."}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "label" in data
        assert "confidence" in data
        assert isinstance(data["label"], str)
        assert 0 <= data["confidence"] <= 1
    
    @pytest.mark.skipif(
        not MODEL_DIR.exists(),
        reason="Model not trained yet - run training first"
    )
    def test_predict_returns_valid_label(self):
        """Prediction should return a valid document type label."""
        response = client.post(
            "/predict",
            json={"text": "Invoice #INV-2024-999 for software development services. Total: $5,000"}
        )
        assert response.status_code == 200
        data = response.json()
        # With a small demo dataset, we don't assert specific labels,
        # but the returned label should be one of the known types
        valid_labels = ["invoice", "purchase_order", "bank_statement", "tax_notice", "contract", "other"]
        assert data["label"] in valid_labels


class TestModelLoading:
    """Tests for model loading behavior."""
    
    def test_artifacts_device_is_set(self):
        """Artifacts should have a device set."""
        assert artifacts.device is not None
    
    def test_model_dir_path_is_absolute(self):
        """Model directory path should be absolute."""
        assert MODEL_DIR.is_absolute()

