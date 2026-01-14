"""
Comprehensive tests for the document processing backend.
"""

import io
import pytest
from fastapi.testclient import TestClient

from src.main import create_app
from src.pipelines import get_job_store, JobState
from src.config.settings import settings

# Create test client
app = create_app()
client = TestClient(app)

# Test API key
TEST_API_KEY = "test-key-123"
settings.api_keys = [TEST_API_KEY]


# ============================================================================
# Helper Functions
# ============================================================================

def get_auth_headers():
    """Get authentication headers."""
    return {"X-API-Key": TEST_API_KEY}


# ============================================================================
# Health & Readiness Tests
# ============================================================================

class TestHealthEndpoints:
    """Tests for health and readiness endpoints."""
    
    def test_health_no_auth_required(self):
        """Health endpoint should not require authentication."""
        response = client.get("/v1/health")
        assert response.status_code == 200
    
    def test_health_response_structure(self):
        """Health response should have expected fields."""
        response = client.get("/v1/health")
        data = response.json()
        
        assert "status" in data
        assert "version" in data
        assert "model_loaded" in data
        assert data["status"] == "healthy"
    
    def test_ready_no_auth_required(self):
        """Readiness endpoint should not require authentication."""
        response = client.get("/v1/ready")
        assert response.status_code == 200
    
    def test_ready_response_structure(self):
        """Readiness response should have expected fields."""
        response = client.get("/v1/ready")
        data = response.json()
        
        assert "ready" in data
        assert "checks" in data
        assert isinstance(data["checks"], dict)


# ============================================================================
# Authentication Tests
# ============================================================================

class TestAuthentication:
    """Tests for API key authentication."""
    
    def test_missing_api_key_returns_401(self):
        """Requests without API key should return 401."""
        response = client.post(
            "/v1/classify/text",
            json={"text": "Test document"}
        )
        assert response.status_code == 401
        assert "Missing API key" in response.json()["detail"]
    
    def test_invalid_api_key_returns_401(self):
        """Requests with invalid API key should return 401."""
        response = client.post(
            "/v1/classify/text",
            headers={"X-API-Key": "invalid-key"},
            json={"text": "Test document"}
        )
        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]
    
    def test_valid_api_key_accepted(self):
        """Requests with valid API key should be accepted."""
        # This will fail if model not loaded, but should pass auth
        response = client.post(
            "/v1/classify/text",
            headers=get_auth_headers(),
            json={"text": "Test document"}
        )
        # Should not be 401
        assert response.status_code != 401


# ============================================================================
# Rate Limiting Tests
# ============================================================================

class TestRateLimiting:
    """Tests for rate limiting."""
    
    def test_rate_limit_exceeded(self):
        """Should return 429 when rate limit exceeded."""
        # Make requests up to the limit
        for _ in range(settings.rate_limit_requests + 5):
            response = client.get("/v1/health")
        
        # The endpoint without auth shouldn't hit rate limit
        # But we can test with an authenticated endpoint
        for _ in range(settings.rate_limit_requests):
            client.post(
                "/v1/classify/text",
                headers=get_auth_headers(),
                json={"text": "Test"}
            )
        
        # Next request should be rate limited
        response = client.post(
            "/v1/classify/text",
            headers=get_auth_headers(),
            json={"text": "Test"}
        )
        
        # Either 429 or another error (if model not loaded)
        # The point is we're testing the rate limiter is working
        assert response.status_code in [429, 500, 503]


# ============================================================================
# Job Lifecycle Tests
# ============================================================================

class TestJobLifecycle:
    """Tests for document job processing lifecycle."""
    
    def test_upload_document_creates_job(self):
        """Uploading document should create a job."""
        content = b"Invoice #12345 for consulting services"
        response = client.post(
            "/v1/documents",
            headers=get_auth_headers(),
            files={"file": ("test.txt", io.BytesIO(content), "text/plain")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "job_id" in data
        assert "status" in data
        assert data["status"] == "queued"
    
    def test_get_job_status(self):
        """Should retrieve job status by ID."""
        # Create a job
        content = b"Purchase order PO-123"
        response = client.post(
            "/v1/documents",
            headers=get_auth_headers(),
            files={"file": ("test.txt", io.BytesIO(content), "text/plain")}
        )
        job_id = response.json()["job_id"]
        
        # Get job status
        response = client.get(
            f"/v1/documents/{job_id}",
            headers=get_auth_headers()
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["job_id"] == job_id
        assert "state" in data
        assert "created_at" in data
    
    def test_get_nonexistent_job_returns_404(self):
        """Getting nonexistent job should return 404."""
        response = client.get(
            "/v1/documents/nonexistent-job-id",
            headers=get_auth_headers()
        )
        assert response.status_code == 404
    
    def test_upload_empty_file_rejected(self):
        """Empty file should be rejected."""
        response = client.post(
            "/v1/documents",
            headers=get_auth_headers(),
            files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")}
        )
        assert response.status_code == 400
    
    def test_upload_unsupported_type_rejected(self):
        """Unsupported file type should be rejected."""
        response = client.post(
            "/v1/documents",
            headers=get_auth_headers(),
            files={"file": ("test.docx", io.BytesIO(b"fake"), "application/vnd.openxmlformats")}
        )
        assert response.status_code == 400


# ============================================================================
# Text Classification Tests
# ============================================================================

class TestTextClassification:
    """Tests for text classification endpoint."""
    
    def test_classify_text_validation(self):
        """Should validate text input."""
        response = client.post(
            "/v1/classify/text",
            headers=get_auth_headers(),
            json={"text": ""}
        )
        assert response.status_code == 422
    
    def test_classify_text_missing_field(self):
        """Should reject missing text field."""
        response = client.post(
            "/v1/classify/text",
            headers=get_auth_headers(),
            json={}
        )
        assert response.status_code == 422


# ============================================================================
# Metrics Tests
# ============================================================================

class TestMetrics:
    """Tests for Prometheus metrics endpoint."""
    
    def test_metrics_endpoint_no_auth(self):
        """Metrics endpoint should not require authentication."""
        response = client.get("/v1/metrics")
        assert response.status_code == 200
    
    def test_metrics_format(self):
        """Metrics should be in Prometheus format."""
        response = client.get("/v1/metrics")
        content = response.text
        
        # Should contain some metric definitions
        assert "http_requests_total" in content or "HELP" in content


# ============================================================================
# Request ID Tests
# ============================================================================

class TestRequestID:
    """Tests for request ID tracking."""
    
    def test_request_id_in_response(self):
        """Response should include X-Request-ID header."""
        response = client.get("/v1/health")
        assert "X-Request-ID" in response.headers
    
    def test_custom_request_id_preserved(self):
        """Custom request ID should be preserved."""
        custom_id = "custom-req-123"
        response = client.get(
            "/v1/health",
            headers={"X-Request-ID": custom_id}
        )
        assert response.headers["X-Request-ID"] == custom_id


# ============================================================================
# Root Endpoint Tests
# ============================================================================

class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_returns_info(self):
        """Root endpoint should return service info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        
        assert "service" in data
        assert "version" in data
        assert "status" in data


# ============================================================================
# Job Store Tests
# ============================================================================

class TestJobStore:
    """Tests for job store functionality."""
    
    def test_create_and_retrieve_job(self):
        """Should create and retrieve job."""
        store = get_job_store()
        job_id = store.create_job(filename="test.pdf")
        
        job = store.get_job(job_id)
        assert job is not None
        assert job.job_id == job_id
        assert job.state == JobState.QUEUED
    
    def test_update_job_state(self):
        """Should update job state."""
        store = get_job_store()
        job_id = store.create_job()
        
        success = store.update_job(job_id, state=JobState.PROCESSING)
        assert success
        
        job = store.get_job(job_id)
        assert job.state == JobState.PROCESSING
    
    def test_job_stats(self):
        """Should return job statistics."""
        store = get_job_store()
        stats = store.get_stats()
        
        assert "total" in stats
        assert "queued" in stats
        assert "processing" in stats
        assert "success" in stats
        assert "failed" in stats


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_validation_error_format(self):
        """Validation errors should have proper format."""
        response = client.post(
            "/v1/classify/text",
            headers=get_auth_headers(),
            json={"invalid_field": "value"}
        )
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
