#!/bin/bash

# Quick start script for Document Processing Backend Service

set -e

echo "🚀 Document Processing Backend Service - Quick Start"
echo "===================================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
echo "✓ Dependencies installed"
echo ""

# Check Tesseract
echo "🔍 Checking Tesseract OCR..."
if command -v tesseract &> /dev/null; then
    tesseract_version=$(tesseract --version 2>&1 | head -n 1)
    echo "✓ $tesseract_version"
else
    echo "⚠️  Tesseract not found. Install it for OCR support:"
    echo "   macOS:  brew install tesseract"
    echo "   Ubuntu: sudo apt-get install tesseract-ocr"
fi
echo ""

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cat > .env << 'EOF'
# API Keys (comma-separated)
API_KEYS=dev-key-12345

# Rate Limiting
RATE_LIMIT_REQUESTS=60
RATE_LIMIT_WINDOW=60

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Server
HOST=0.0.0.0
PORT=8000
EOF
    echo "✓ .env file created with defaults"
else
    echo "✓ .env file already exists"
fi
echo ""

# Check if model exists
if [ -d "models/run1" ]; then
    echo "✓ Model found at models/run1"
else
    echo "⚠️  Model not found. Train a model first:"
    echo "   python -m src.train --data data/sample_dataset.csv --output_dir models/run1 --epochs 10"
fi
echo ""

# Prompt to start service
echo "✅ Setup complete!"
echo ""
echo "To start the service, run:"
echo "   python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "Or run directly:"
echo "   python src/main.py"
echo ""
echo "📚 API Documentation will be available at:"
echo "   http://localhost:8000/docs"
echo ""
