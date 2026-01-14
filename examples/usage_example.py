"""
Example usage of the Document Processing Backend API.

Run this script to see how to interact with the API.
"""

import requests
import json
import time
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = "dev-key-12345"  # Change to your API key

# Headers
HEADERS = {
    "X-API-Key": API_KEY
}


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def check_health():
    """Check API health status."""
    print_section("Health Check")
    
    response = requests.get(f"{API_BASE_URL}/v1/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    return response.status_code == 200


def check_readiness():
    """Check API readiness."""
    print_section("Readiness Check")
    
    response = requests.get(f"{API_BASE_URL}/v1/ready")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(json.dumps(data, indent=2))
    
    return data.get("ready", False)


def classify_text(text):
    """Classify text directly (synchronous)."""
    print_section("Text Classification (Sync)")
    
    payload = {"text": text}
    response = requests.post(
        f"{API_BASE_URL}/v1/classify/text",
        headers=HEADERS,
        json=payload
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"\nPredicted Label: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Confidence Level: {result['confidence_level']}")
        print(f"Needs Review: {result['needs_review']}")
        
        print("\nTop Predictions:")
        for pred in result['top_predictions']:
            print(f"  {pred['rank']}. {pred['label']}: {pred['probability']:.2%}")
        
        print("\nUncertainty Metrics:")
        print(f"  Entropy: {result['uncertainty_metrics']['entropy']:.4f}")
        print(f"  Margin: {result['uncertainty_metrics']['margin']:.4f}")
    else:
        print(f"Error: {response.text}")


def upload_document(file_path):
    """Upload a document for async processing."""
    print_section(f"Upload Document: {file_path}")
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return None
    
    with open(file_path, 'rb') as f:
        files = {'file': (Path(file_path).name, f)}
        response = requests.post(
            f"{API_BASE_URL}/v1/documents",
            headers=HEADERS,
            files=files
        )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Job ID: {result['job_id']}")
        print(f"Status: {result['status']}")
        print(f"Message: {result['message']}")
        return result['job_id']
    else:
        print(f"Error: {response.text}")
        return None


def get_job_status(job_id):
    """Get job status and result."""
    response = requests.get(
        f"{API_BASE_URL}/v1/documents/{job_id}",
        headers=HEADERS
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.text}")
        return None


def wait_for_job(job_id, max_wait=60):
    """Wait for job to complete and display result."""
    print_section(f"Waiting for Job: {job_id}")
    
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        job = get_job_status(job_id)
        
        if not job:
            return None
        
        print(f"State: {job['state']} (elapsed: {int(time.time() - start_time)}s)", end='\r')
        
        if job['state'] == 'SUCCESS':
            print("\n\n✅ Job completed successfully!\n")
            result = job['result']
            
            print(f"Predicted Label: {result['predicted_label']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Confidence Level: {result['confidence_level']}")
            print(f"Needs Review: {result['needs_review']}")
            print(f"Processing Time: {job['processing_time_ms']}ms")
            
            print("\nExtracted Text:")
            print(f"  Length: {result['extracted_text_length']} characters")
            print(f"  Preview: {result['extracted_text_preview'][:100]}...")
            
            print("\nTop Predictions:")
            for pred in result['top_predictions']:
                print(f"  {pred['rank']}. {pred['label']}: {pred['probability']:.2%}")
            
            return job
        
        elif job['state'] == 'FAILED':
            print("\n\n❌ Job failed!\n")
            print(f"Error: {job.get('error', 'Unknown error')}")
            return job
        
        time.sleep(1)
    
    print("\n\n⏱️  Timeout waiting for job")
    return None


def main():
    """Run example usage."""
    print("\n" + "="*60)
    print("  Document Processing Backend API - Example Usage")
    print("="*60)
    
    # Check if API is running
    if not check_health():
        print("\n❌ API is not running. Start it with:")
        print("   python -m uvicorn src.main:app --reload")
        return
    
    # Check if model is ready
    if not check_readiness():
        print("\n⚠️  Model not loaded. Train a model first:")
        print("   python -m src.train --data data/sample_dataset.csv --output_dir models/run1")
        return
    
    # Example 1: Classify text
    classify_text("Invoice #INV-2024-001 for consulting services rendered in January. Total amount due: $5,250.00")
    
    # Example 2: Upload a document (create a test file if needed)
    test_file = "test_document.txt"
    if not Path(test_file).exists():
        with open(test_file, 'w') as f:
            f.write("Purchase Order PO-12345\n\nItem: Office Supplies\nQuantity: 100\nTotal: $1,250.00")
    
    job_id = upload_document(test_file)
    
    if job_id:
        wait_for_job(job_id)
    
    print("\n" + "="*60)
    print("  Examples Complete!")
    print("="*60)
    print("\n📚 For more information, visit: http://localhost:8000/docs")
    print()


if __name__ == "__main__":
    main()
