# Financial Document Classifier

An end-to-end machine learning system for classifying financial documents using a fine-tuned DistilBERT model, with **PDF/OCR support** and **uncertainty quantification** for production-ready predictions.

## 🎯 Problem Statement

Organizations process thousands of financial documents daily. Manual classification is slow, error-prone, and expensive. This system automates document classification with:

- **98% accuracy** on 6 document types
- **Real document support** (PDFs, scanned images via OCR)
- **Uncertainty quantification** (flags low-confidence predictions for human review)
- **Production-ready API** with file upload support

## ✨ Key Features

### 1. Multi-Format Document Processing
- **PDF files** - Native text extraction + OCR fallback for scanned documents
- **Image files** - PNG, JPG, TIFF support via Tesseract OCR
- **Plain text** - Direct text classification

### 2. Uncertainty Quantification
Not all predictions are equal. The system provides:
- **Confidence scores** with calibrated probabilities
- **Human review flags** for ambiguous predictions
- **Top-k predictions** to see alternative classifications
- **Entropy & margin metrics** for uncertainty analysis

### 3. Supported Document Types
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
| **Model** | DistilBERT (Hugging Face Transformers) |
| **Framework** | PyTorch |
| **API** | FastAPI + Uvicorn |
| **PDF Processing** | PyMuPDF |
| **OCR** | Tesseract (pytesseract) |
| **ML Utilities** | scikit-learn, pandas, NumPy |

## 📁 Project Structure

```
financial_doc_classifier/
├── api/
│   └── app.py              # FastAPI with /predict and /predict/document
├── src/
│   ├── dataset.py          # Data loading utilities
│   ├── model.py            # Model creation
│   ├── train.py            # Training pipeline
│   ├── ocr.py              # PDF/image text extraction
│   └── confidence.py       # Uncertainty quantification
├── tests/
│   ├── test_api.py         # API endpoint tests
│   ├── test_dataset.py     # Dataset utility tests
│   ├── test_ocr.py         # OCR module tests
│   └── test_confidence.py  # Confidence module tests
├── data/
│   └── sample_dataset.csv  # Training data (298 samples)
├── models/                  # Trained model artifacts
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Tesseract OCR (for image/scanned PDF support)

### Installation

```bash
# Clone the repository
git clone https://github.com/ARSHKHAN98/financial_doc_classifier.git
cd financial_doc_classifier

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Tesseract (for OCR support)
# macOS: brew install tesseract
# Ubuntu: sudo apt-get install tesseract-ocr
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

### Train the Model

```bash
python -m src.train \
    --data data/sample_dataset.csv \
    --output_dir models/run1 \
    --epochs 10 \
    --batch_size 8
```

### Start the API

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

## 📡 API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Classify Text
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Invoice #INV-2024-001 for consulting services. Total: $5,250.00"}'
```

**Response:**
```json
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

### Classify Document File (PDF/Image)
```bash
curl -X POST http://localhost:8000/predict/document \
  -F "file=@/path/to/invoice.pdf"
```

**Response includes:**
- All fields from text classification
- `extracted_text_preview` - First 200 characters of extracted text
- `text_length` - Total extracted text length

### Understanding the Response

| Field | Description |
|-------|-------------|
| `predicted_label` | Top predicted document type |
| `confidence` | Probability of prediction (0-1) |
| `confidence_level` | Human-readable level (very_high/high/medium/low/very_low) |
| `needs_review` | `true` if human review is recommended |
| `review_reason` | Explanation for review flag |
| `top_predictions` | Top-3 predictions with probabilities |
| `uncertainty_metrics.entropy` | Higher = more uncertain |
| `uncertainty_metrics.margin` | Difference between top-1 and top-2 |

### When is Human Review Flagged?

The system flags predictions for review when:
- Confidence < 50%
- Margin between top predictions < 15%
- High entropy (uncertainty spread across classes)
- Borderline confidence with ambiguity

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_confidence.py -v

# Run with coverage
pytest tests/ -v --cov=src --cov=api
```

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 98% |
| **Dataset Size** | 298 samples |
| **Classes** | 6 |
| **Model Size** | ~250 MB |
| **Inference Time** | ~50ms (CPU) |

## 🔧 Configuration

### Confidence Thresholds

Located in `src/confidence.py`:

```python
class ConfidenceThresholds:
    LOW_CONFIDENCE = 0.5      # Below: definitely needs review
    HIGH_CONFIDENCE = 0.85    # Above: high confidence
    MIN_MARGIN = 0.15         # Minimum margin between top-2
```

### Training Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | required | Path to CSV dataset |
| `--output_dir` | required | Model save directory |
| `--epochs` | 3 | Training epochs |
| `--batch_size` | 4 | Batch size |
| `--lr` | 2e-5 | Learning rate |
| `--max_length` | 256 | Max sequence length |

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Document      │────▶│   OCR/PDF    │────▶│   DistilBERT    │
│   (PDF/Image)   │     │   Extraction │     │   Classifier    │
└─────────────────┘     └──────────────┘     └────────┬────────┘
                                                      │
                        ┌──────────────┐              │
                        │  Uncertainty │◀─────────────┘
                        │  Analysis    │
                        └──────┬───────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        ┌──────────┐    ┌──────────┐    ┌──────────┐
        │ Label    │    │ Confidence│    │ Review   │
        │ Prediction│    │ Score    │    │ Flag     │
        └──────────┘    └──────────┘    └──────────┘
```

## 🎓 Why This Matters

### For Industry
- **Accounts Payable Automation** - Auto-route invoices to correct workflow
- **Compliance** - Audit trails with confidence scores
- **Cost Reduction** - Process 1000s of documents/hour vs. dozens manually

### For Learning
- **End-to-end ML pipeline** - Not just a notebook experiment
- **Production patterns** - Error handling, logging, testing
- **Modern stack** - Transformers + FastAPI is industry standard
- **Uncertainty quantification** - Critical for real-world deployment

## ⚠️ Limitations

- **English only** - Model trained on English documents
- **Sample dataset** - 298 samples; production needs more data
- **OCR quality** - Depends on document scan quality
- **No layout analysis** - Uses text only, not visual structure

## 🚧 Future Improvements

- [ ] Multi-language support (German, French)
- [ ] Layout-aware classification using document images
- [ ] Active learning for continuous improvement
- [ ] Docker containerization
- [ ] Batch processing endpoint
- [ ] SHAP/attention visualization for explainability

## 📄 License

MIT License - feel free to use and modify for your projects.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [PyMuPDF](https://pymupdf.readthedocs.io/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
