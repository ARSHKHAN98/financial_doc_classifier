# Financial Document Classifier

A machine learning project that classifies financial documents (invoices, purchase orders, bank statements, etc.) using a fine-tuned DistilBERT model, exposed through a FastAPI REST service.

## Overview

This project demonstrates an end-to-end ML pipeline for text classification:

1. **Data Processing** – Load and split labeled document data
2. **Model Training** – Fine-tune DistilBERT for multi-class classification
3. **Evaluation** – Generate classification reports and confusion matrices
4. **Serving** – REST API for real-time predictions

### Supported Document Types

- `invoice` – Billing documents for goods/services
- `purchase_order` – Procurement requests
- `bank_statement` – Account transaction summaries
- `tax_notice` – Tax-related notifications
- `contract` – Legal agreements
- `other` – Miscellaneous documents

## Tech Stack

- **Model**: [DistilBERT](https://huggingface.co/distilbert-base-uncased) (Hugging Face Transformers)
- **Framework**: PyTorch
- **API**: FastAPI + Uvicorn
- **ML Utilities**: scikit-learn, pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

## Project Structure

```
financial_doc_classifier/
├── api/
│   └── app.py              # FastAPI application
├── data/
│   └── sample_dataset.csv  # Sample training data
├── models/                  # Saved model artifacts (after training)
│   └── run1/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       ├── label_encoder.npy
│       ├── metrics.json
│       └── confusion_matrix.png
├── src/
│   ├── __init__.py
│   ├── dataset.py          # Data loading utilities
│   ├── model.py            # Model creation
│   └── train.py            # Training script
├── tests/
│   ├── test_api.py         # API endpoint tests
│   └── test_dataset.py     # Dataset utility tests
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone or extract the project
cd financial_doc_classifier

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Training

Train the model using the provided sample dataset or your own labeled data:

```bash
python -m src.train \
    --data data/sample_dataset.csv \
    --output_dir models/run1 \
    --epochs 3 \
    --batch_size 4
```

### Training Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | required | Path to CSV with `text` and `label` columns |
| `--output_dir` | required | Directory to save model artifacts |
| `--epochs` | 3 | Number of training epochs |
| `--batch_size` | 4 | Training batch size |
| `--lr` | 2e-5 | Learning rate |
| `--max_length` | 256 | Maximum sequence length |

### Training Output

After training, you'll see:
- Classification report with precision, recall, F1-score per class
- Saved model files in `output_dir`
- `metrics.json` with detailed evaluation metrics
- `confusion_matrix.png` visualization

Example output:
```
2024-01-15 10:30:45 [INFO] Using device: cpu
2024-01-15 10:30:45 [INFO] Loading dataset from data/sample_dataset.csv
2024-01-15 10:30:45 [INFO] Train samples: 19, Validation samples: 5
2024-01-15 10:30:45 [INFO] Labels: ['bank_statement', 'contract', 'invoice', 'other', 'purchase_order', 'tax_notice']
2024-01-15 10:30:46 [INFO] Starting training for 3 epochs...
2024-01-15 10:31:12 [INFO] Epoch 1/3 - Loss: 1.8234
2024-01-15 10:31:38 [INFO] Epoch 2/3 - Loss: 1.2156
2024-01-15 10:32:04 [INFO] Epoch 3/3 - Loss: 0.7823
2024-01-15 10:32:05 [INFO] Training complete!
```

## Running the API

Start the FastAPI server:

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

### API Endpoints

#### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "/path/to/models/run1"
}
```

#### Predict Document Type

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Invoice #INV-2024-001 for consulting services. Total amount due: $5,250.00"}'
```

Response:
```json
{
  "label": "invoice",
  "confidence": 0.9234
}
```

### Interactive API Docs

FastAPI provides automatic interactive documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov=api
```

Note: Some tests require a trained model. Run training first for full test coverage.

## Using Your Own Data

1. Prepare a CSV file with `text` and `label` columns:
   ```csv
   text,label
   "Your document text here","category_name"
   ```

2. Ensure you have enough samples per class (minimum 3-4 for stratified splitting)

3. Run training with your dataset:
   ```bash
   python -m src.train --data path/to/your/data.csv --output_dir models/custom
   ```

4. Update `MODEL_DIR` in `api/app.py` if using a different model path

## Limitations

- **Small sample dataset**: The included `sample_dataset.csv` is for demonstration only. Real-world performance requires more training data.
- **English only**: The model is based on `distilbert-base-uncased`, optimized for English text.
- **CPU inference**: GPU support exists but requires CUDA setup.
- **No data preprocessing**: Raw text is tokenized directly; advanced cleaning may improve results.

## Future Improvements

- [ ] Add data augmentation for small datasets
- [ ] Implement confidence thresholding for uncertain predictions
- [ ] Add batch prediction endpoint
- [ ] Containerize with Docker
- [ ] Add model versioning and A/B testing support
- [ ] Implement MLflow for experiment tracking

## License

MIT License – feel free to use and modify for your projects.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for the DistilBERT implementation
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
