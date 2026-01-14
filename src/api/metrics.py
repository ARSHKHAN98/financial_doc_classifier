"""
Prometheus metrics for observability.
"""

from prometheus_client import Counter, Histogram, Gauge
from enum import Enum


class MetricLabels(str, Enum):
    """Standard metric labels."""
    ENDPOINT = "endpoint"
    METHOD = "method"
    STATUS = "status"
    JOB_STATE = "job_state"


# Request metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Job metrics
jobs_total = Counter(
    "jobs_total",
    "Total jobs created",
    ["state"]
)

job_processing_duration_seconds = Histogram(
    "job_processing_duration_seconds",
    "Job processing duration in seconds",
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0]
)

jobs_active = Gauge(
    "jobs_active",
    "Number of currently active jobs"
)

# ML Model metrics
model_inference_duration_seconds = Histogram(
    "model_inference_duration_seconds",
    "Model inference duration in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

predictions_total = Counter(
    "predictions_total",
    "Total predictions made",
    ["predicted_label", "needs_review"]
)

# OCR metrics
ocr_extraction_duration_seconds = Histogram(
    "ocr_extraction_duration_seconds",
    "OCR text extraction duration in seconds",
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
)

# Error metrics
errors_total = Counter(
    "errors_total",
    "Total errors",
    ["error_type"]
)
