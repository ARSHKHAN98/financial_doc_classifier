# Financial Document Classifier

An end-to-end machine learning system for classifying financial documents using a fine-tuned DistilBERT model, with **PDF/OCR support**, **uncertainty quantification**, and **robust evaluation methodology**.

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

### 3. Robust Training Pipeline
- **Stratified splitting** - Maintains class distribution in train/val/test
- **K-fold cross-validation** - More reliable accuracy estimates
- **Class weight balancing** - Handles imbalanced datasets
- **Reproducible** - Fixed random seeds

### 4. Supported Document Types
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
│   ├── train.py            # Training pipeline with CV support
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

## 🎓 Training Options

### Standard Training (Train/Val Split)

```bash
python -m src.train \
    --data data/sample_dataset.csv \
    --output_dir models/run1 \
    --epochs 10 \
    --batch_size 8
```

### With K-Fold Cross-Validation (Recommended for Small Datasets)

```bash
python -m src.train \
    --data data/sample_dataset.csv \
    --output_dir models/run1 \
    --epochs 10 \
    --cross_validate \
    --n_folds 5
```

### With Class Weight Balancing (For Imbalanced Data)

```bash
python -m src.train \
    --data data/sample_dataset.csv \
    --output_dir models/run1 \
    --epochs 10 \
    --use_class_weights
```

### With Held-Out Test Set (3-Way Split)

```bash
python -m src.train \
    --data data/sample_dataset.csv \
    --output_dir models/run1 \
    --epochs 10 \
    --holdout_test
```

### All Training Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | required | Path to CSV dataset |
| `--output_dir` | required | Model save directory |
| `--epochs` | 3 | Training epochs |
| `--batch_size` | 8 | Batch size |
| `--lr` | 2e-5 | Learning rate |
| `--max_length` | 256 | Max sequence length |
| `--seed` | 42 | Random seed for reproducibility |
| `--test_size` | 0.2 | Validation split ratio |
| `--cross_validate` | false | Enable k-fold cross-validation |
| `--n_folds` | 5 | Number of CV folds |
| `--use_class_weights` | false | Balance classes with weights |
| `--holdout_test` | false | Use train/val/test split |

## 📊 Evaluation Methodology

### Data Splitting Strategy

1. **Stratified Splitting**: All splits maintain the original class distribution
2. **Random Seed**: Fixed seed (42) ensures reproducibility
3. **Split Options**:
   - Train/Val (80/20) - Default
   - Train/Val/Test (60/20/20) - With `--holdout_test`
   - K-Fold CV - With `--cross_validate`

### Cross-Validation

For small datasets, k-fold cross-validation provides more robust accuracy estimates:

```
5-Fold CV Results:
├── Fold 1: 96.67%
├── Fold 2: 98.33%
├── Fold 3: 100.00%
├── Fold 4: 98.33%
└── Fold 5: 96.67%

Mean Accuracy: 98.00% (+/- 1.25%)
```

### Handling Class Imbalance

The `--use_class_weights` flag computes balanced weights:

```
weight_i = n_samples / (n_classes × n_samples_i)
```

This penalizes misclassification of minority classes more heavily.

## 📡 API Endpoints

### Start the API

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

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

## 🧪 Running Tests

```bash
# Run all tests (59 tests)
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov=api
```

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

## ⚠️ Limitations & Honest Assessment

### Dataset Size
- **298 samples** is small for production ML
- This is a **prototype/proof-of-concept** demonstrating the full pipeline
- For production: 1,000+ samples per class recommended

### What This Project Demonstrates
- ✅ End-to-end ML pipeline design (data → model → API)
- ✅ Production engineering patterns (logging, testing, error handling)
- ✅ Proper evaluation methodology (stratified splits, cross-validation)
- ✅ Uncertainty quantification for real-world deployment
- ✅ Document processing with OCR

### What Would Be Needed for Production
- [ ] Larger, real-world dataset
- [ ] More extensive hyperparameter tuning
- [ ] Model compression/optimization
- [ ] Monitoring and drift detection
- [ ] A/B testing infrastructure

### Other Limitations
- **English only** - Model trained on English documents
- **OCR quality** - Depends on document scan quality
- **No layout analysis** - Uses text only, not visual structure

## 🚧 Future Improvements

- [ ] Multi-language support (German, French)
- [ ] Layout-aware classification using document images
- [ ] Active learning for continuous improvement
- [ ] Docker containerization
- [ ] Batch processing endpoint
- [ ] SHAP/attention visualization for explainability
- [ ] MLflow for experiment tracking

## 📄 License

MIT License - feel free to use and modify for your projects.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [PyMuPDF](https://pymupdf.readthedocs.io/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
