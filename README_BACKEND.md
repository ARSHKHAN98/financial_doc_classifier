# Document Processing Backend Service

A **production-grade document processing backend** built with FastAPI. Handles file ingestion, async job processing, OCR, and ML-powered document classification with enterprise-grade features like authentication, rate limiting, and observability.

## 🎯 Overview

This is a **backend platform**, not just an ML model. It provides:

- **Async Job Processing**: Upload documents, get job IDs, poll for results
- **File Ingestion Pipeline**: Validates, extracts text (OCR), and classifies documents
- **API-First Design**: RESTful API with OpenAPI documentation
- **Production Features**: Auth, rate limiting, metrics, health checks, structured logging
- **Reliability**: Timeouts, error handling, request tracking, graceful degradation

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Applications                       │
│              (Web Apps, Mobile Apps, Other Services)             │
└────────────────────────┬────────────────────────────────────────┘
                         │ X-API-Key Header
                         │ Rate Limited
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Application                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   API Layer  │  │   Pipelines  │  │   Services Layer     │  │
│  │   /v1/...    │──│  Job Manager │──│  Ingestion | OCR     │  │
│  │   Routes     │  │  Processor   │  │  Classification      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│         │                  │                      │              │
│         │                  │                      ▼              │
│         │                  │            ┌──────────────────┐    │
│         │                  │            │   ML Models      │    │
│         │                  │            │   (DistilBERT)   │    │
│         │                  │            └──────────────────┘    │
│         ▼                  ▼                                     │
│  ┌──────────────┐  ┌──────────────┐                            │
│  │  Middleware  │  │  Job Store   │                            │
│  │  Auth, Rate  │  │  (In-Memory) │                            │
│  │  Limit, Logs │  │              │                            │
│  └──────────────┘  └──────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

## 📂 Project Structure

```
financial_doc_classifier/
├── src/
│   ├── api/                    # API layer
│   │   ├── routes.py           # API endpoints (/v1/...)
│   │   ├── schemas.py          # Pydantic request/response models
│   │   ├── auth.py             # API key auth + rate limiting
│   │   └── metrics.py          # Prometheus metrics
│   ├── services/               # Business logic services
│   │   ├── ingestion.py        # File validation
│   │   ├── ocr_service.py      # Text extraction wrapper
│   │   └── classification.py   # ML classification wrapper
│   ├── pipelines/              # Processing workflows
│   │   ├── job_manager.py      # Job state management
│   │   └── document_processor.py  # Main processing pipeline
│   ├── models/                 # ML model wrappers
│   │   └── inference.py        # Model loading & inference
│   ├── config/                 # Configuration management
│   │   └── settings.py         # Settings from env vars
│   ├── utils/                  # Utilities
│   │   └── logging_config.py   # Structured JSON logging
│   ├── main.py                 # FastAPI app factory
│   ├── ocr.py                  # OCR implementation (existing)
│   ├── confidence.py           # Uncertainty quantification (existing)
│   └── ...                     # Other modules
├── tests/
│   ├── test_backend.py         # Backend API tests
│   └── ...                     # Other tests
├── models/                     # Trained ML models
│   └── run1/
├── data/                       # Training data
├── monitoring/                 # Monitoring configs
│   └── prometheus.yml
├── Dockerfile                  # Production Docker image
├── docker-compose.yml          # Full stack deployment
├── requirements.txt            # Python dependencies
├── .env.example                # Example environment variables
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Tesseract OCR (for document text extraction)
- Docker & Docker Compose (optional, for containerized deployment)

### Installation

```bash
# Clone repository
git clone <repo-url>
cd financial_doc_classifier

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Tesseract OCR
# macOS: brew install tesseract
# Ubuntu: sudo apt-get install tesseract-ocr
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

### Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
# Important: Change API_KEYS in production!
```

### Train Model (First Time)

```bash
# Train the ML model
python -m src.train \
    --data data/sample_dataset.csv \
    --output_dir models/run1 \
    --epochs 10
```

### Run Service

```bash
# Development mode
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Or directly
python src/main.py

# Production mode (with Gunicorn)
gunicorn src.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# With monitoring stack (Prometheus + Grafana)
docker-compose --profile monitoring up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

## 📡 API Documentation

### Base URL

```
http://localhost:8000
```

### Authentication

All endpoints (except `/health`, `/ready`, `/metrics`) require an API key:

```bash
X-API-Key: your-api-key-here
```

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

#### Health & Observability

```bash
# Health check (no auth)
GET /v1/health

# Readiness check (no auth)
GET /v1/ready

# Prometheus metrics (no auth)
GET /v1/metrics
```

#### Document Processing (Async)

```bash
# Upload document for processing
POST /v1/documents
Headers: X-API-Key: <key>
Body: multipart/form-data with 'file' field

Response:
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Document queued for processing..."
}

# Get job status and result
GET /v1/documents/{job_id}
Headers: X-API-Key: <key>

Response:
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "state": "SUCCESS",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:05Z",
  "filename": "invoice.pdf",
  "processing_time_ms": 1234,
  "result": {
    "predicted_label": "invoice",
    "confidence": 0.9823,
    "confidence_level": "very_high",
    "needs_review": false,
    "top_predictions": [...],
    "uncertainty_metrics": {...},
    "extracted_text_length": 456,
    "extracted_text_preview": "Invoice #INV-2024..."
  }
}
```

#### Text Classification (Sync)

```bash
# Classify text directly (synchronous)
POST /v1/classify/text
Headers: X-API-Key: <key>
Body: {
  "text": "Invoice #INV-2024-001 for consulting services. Total: $5,250.00"
}

Response:
{
  "predicted_label": "invoice",
  "confidence": 0.9823,
  "confidence_level": "very_high",
  "needs_review": false,
  "review_reason": null,
  "top_predictions": [
    {"label": "invoice", "probability": 0.9823, "rank": 1},
    {"label": "purchase_order", "probability": 0.0089, "rank": 2},
    {"label": "contract", "probability": 0.0045, "rank": 3}
  ],
  "uncertainty_metrics": {
    "entropy": 0.1234,
    "margin": 0.9734
  }
}
```

### Rate Limits

- Default: **60 requests per minute** per API key
- Returns `429 Too Many Requests` when exceeded
- Response includes `Retry-After` header

### Request Tracking

All responses include `X-Request-ID` header for tracing.

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_backend.py -v

# Run backend tests only
pytest tests/test_backend.py -v -k "Test"
```

## 📊 Observability

### Structured Logging

All logs are output as JSON with consistent fields:

```json
{
  "timestamp": "2024-01-15T10:30:00.123Z",
  "level": "INFO",
  "logger": "src.pipelines.document_processor",
  "message": "Job 550e8400: Completed successfully in 1234ms",
  "request_id": "req-123"
}
```

### Prometheus Metrics

Available at `/v1/metrics`:

- `http_requests_total` - Request count by endpoint, method, status
- `http_request_duration_seconds` - Request latency histogram
- `jobs_total` - Job count by state
- `job_processing_duration_seconds` - Job processing time
- `predictions_total` - Predictions by label and review flag
- `errors_total` - Error count by type

### Health Checks

- `/v1/health` - Basic health status
- `/v1/ready` - Readiness for traffic (checks model loaded)
- Docker healthcheck included in Dockerfile

## ⚙️ Configuration

All configuration via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEYS` | `dev-key-12345` | Comma-separated API keys |
| `RATE_LIMIT_REQUESTS` | `60` | Requests per window |
| `RATE_LIMIT_WINDOW` | `60` | Window size (seconds) |
| `MAX_FILE_SIZE_MB` | `10` | Max upload size |
| `JOB_TIMEOUT_SECONDS` | `120` | Job processing timeout |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FORMAT` | `json` | Log format (json/text) |

## 🔒 Security

- **API Key Authentication**: Required for all protected endpoints
- **Rate Limiting**: Per-key limits prevent abuse
- **File Validation**: Size and type checks on uploads
- **Timeout Protection**: Jobs and requests have timeouts
- **Non-root Container**: Docker runs as non-root user
- **Input Validation**: Pydantic schemas validate all inputs

## 📈 Performance

- **Async Processing**: Non-blocking document processing
- **Background Jobs**: FastAPI BackgroundTasks for async work
- **Connection Pooling**: Efficient resource usage
- **Model Caching**: Model loaded once at startup
- **Batch-ready**: Can scale horizontally with load balancer

## 🐳 Production Deployment

### Docker Deployment

```bash
# Build image
docker build -t doc-processor:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -e API_KEYS="prod-key-1,prod-key-2" \
  -v $(pwd)/models:/app/models:ro \
  --name doc-processor \
  doc-processor:latest
```

### Kubernetes Ready

The service is designed to run in Kubernetes:

- Health and readiness probes
- Graceful shutdown
- 12-factor app compliant
- Stateless (job store can be swapped with Redis)

## 🔄 Extending the Service

### Swap In-Memory Store with Redis

The job store and rate limiter use in-memory storage with interfaces designed for easy Redis replacement:

1. Create Redis implementations of `JobStore` and `RateLimiter`
2. Update `get_job_store()` and `get_rate_limiter()` to return Redis versions
3. Add Redis connection to settings

### Add More Document Types

1. Update training data with new labels
2. Retrain model: `python -m src.train ...`
3. No code changes needed!

### Add Authentication Providers

Implement additional auth mechanisms in `src/api/auth.py`:
- JWT tokens
- OAuth2
- mTLS

## 📋 Document Types Supported

| Type | Description |
|------|-------------|
| `invoice` | Billing documents for goods/services |
| `purchase_order` | Procurement requests |
| `bank_statement` | Account transaction summaries |
| `tax_notice` | Tax-related notifications |
| `contract` | Legal agreements |
| `other` | Miscellaneous documents |

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **API Framework** | FastAPI |
| **Server** | Uvicorn / Gunicorn |
| **ML Model** | DistilBERT (Hugging Face) |
| **ML Framework** | PyTorch |
| **PDF Processing** | PyMuPDF |
| **OCR** | Tesseract |
| **Validation** | Pydantic |
| **Metrics** | Prometheus |
| **Containerization** | Docker |

## 📝 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyMuPDF](https://pymupdf.readthedocs.io/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

---

**Built for production. Ready to scale. Easy to extend.**
