"""
Tests for the FastAPI prediction endpoints.

Run with: pytest tests/test_api.py -v
"""

import io
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
        assert "supported_file_types" in data
        assert "confidence_thresholds" in data
        assert data["status"] == "healthy"
    
    def test_health_includes_file_types(self):
        """Health response should list supported file types."""
        response = client.get("/health")
        data = response.json()
        
        file_types = data["supported_file_types"]
        assert ".pdf" in file_types
        assert ".png" in file_types
        assert ".txt" in file_types


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
    def test_predict_returns_full_response(self):
        """Predict should return complete response with uncertainty metrics."""
        response = client.post(
            "/predict", 
            json={"text": "Invoice #12345 for consulting services. Amount due: $1,000."}
        )
        assert response.status_code == 200
        
        data = response.json()
        
        # Check all required fields
        assert "predicted_label" in data
        assert "confidence" in data
        assert "confidence_level" in data
        assert "needs_review" in data
        assert "top_predictions" in data
        assert "uncertainty_metrics" in data
        
        # Validate types
        assert isinstance(data["predicted_label"], str)
        assert isinstance(data["confidence"], float)
        assert isinstance(data["needs_review"], bool)
        assert isinstance(data["top_predictions"], list)
        
        # Validate ranges
        assert 0 <= data["confidence"] <= 1
        assert len(data["top_predictions"]) >= 1
    
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
        
        valid_labels = ["invoice", "purchase_order", "bank_statement", "tax_notice", "contract", "other"]
        assert data["predicted_label"] in valid_labels
    
    @pytest.mark.skipif(
        not MODEL_DIR.exists(),
        reason="Model not trained yet - run training first"
    )
    def test_predict_top_predictions_ordered(self):
        """Top predictions should be ordered by probability."""
        response = client.post(
            "/predict",
            json={"text": "Purchase order PO-12345 for office supplies"}
        )
        assert response.status_code == 200
        data = response.json()
        
        probs = [p["probability"] for p in data["top_predictions"]]
        assert probs == sorted(probs, reverse=True)
    
    @pytest.mark.skipif(
        not MODEL_DIR.exists(),
        reason="Model not trained yet - run training first"
    )
    def test_predict_uncertainty_metrics(self):
        """Response should include valid uncertainty metrics."""
        response = client.post(
            "/predict",
            json={"text": "Bank statement for account ending in 4521"}
        )
        assert response.status_code == 200
        data = response.json()
        
        metrics = data["uncertainty_metrics"]
        assert "entropy" in metrics
        assert "margin" in metrics
        assert metrics["entropy"] >= 0
        assert 0 <= metrics["margin"] <= 1


class TestDocumentUploadEndpoint:
    """Tests for the /predict/document endpoint."""
    
    def test_upload_rejects_unsupported_type(self):
        """Should reject unsupported file types."""
        # Create a fake .docx file
        file_content = b"fake docx content"
        response = client.post(
            "/predict/document",
            files={"file": ("document.docx", io.BytesIO(file_content), "application/vnd.openxmlformats")}
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]
    
    def test_upload_rejects_empty_file(self):
        """Should reject empty files."""
        response = client.post(
            "/predict/document",
            files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")}
        )
        assert response.status_code == 400
    
    @pytest.mark.skipif(
        not MODEL_DIR.exists(),
        reason="Model not trained yet - run training first"
    )
    def test_upload_text_file(self):
        """Should process plain text file uploads."""
        content = b"Invoice #INV-2024-001 for consulting services. Total: $5,000"
        response = client.post(
            "/predict/document",
            files={"file": ("invoice.txt", io.BytesIO(content), "text/plain")}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "predicted_label" in data
        assert "extracted_text_preview" in data
        assert "text_length" in data
        assert data["text_length"] == len(content)
    
    @pytest.mark.skipif(
        not MODEL_DIR.exists(),
        reason="Model not trained yet - run training first"
    )
    def test_upload_response_includes_preview(self):
        """Response should include text preview."""
        content = b"This is a test document with some content for classification."
        response = client.post(
            "/predict/document",
            files={"file": ("test.txt", io.BytesIO(content), "text/plain")}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["extracted_text_preview"].startswith("This is a test")


class TestModelLoading:
    """Tests for model loading behavior."""
    
    def test_artifacts_device_is_set(self):
        """Artifacts should have a device set."""
        assert artifacts.device is not None
    
    def test_model_dir_path_is_absolute(self):
        """Model directory path should be absolute."""
        assert MODEL_DIR.is_absolute()


class TestConfidenceLevels:
    """Tests for confidence level labels in responses."""
    
    @pytest.mark.skipif(
        not MODEL_DIR.exists(),
        reason="Model not trained yet - run training first"
    )
    def test_confidence_level_is_valid(self):
        """Confidence level should be one of the defined levels."""
        response = client.post(
            "/predict",
            json={"text": "Contract agreement between two parties"}
        )
        assert response.status_code == 200
        
        data = response.json()
        valid_levels = ["very_high", "high", "medium", "low", "very_low"]
        assert data["confidence_level"] in valid_levels
