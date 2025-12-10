"""
Training script for the financial document classifier.

Usage:
    python -m src.train --data data/sample_dataset.csv --output_dir models/run1
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from .dataset import load_dataset
from .model import create_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """PyTorch Dataset for tokenized text classification."""
    
    def __init__(self, texts: list, labels: list, tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def train_epoch(model, dataloader: DataLoader, optimizer, scheduler, device) -> float:
    """Run one training epoch and return average loss."""
    model.train()
    total_loss = 0.0
    
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate_model(model, dataloader: DataLoader, device) -> tuple:
    """Evaluate model and return (true_labels, predicted_labels)."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"]
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits.cpu().numpy()
            pred_labels = np.argmax(logits, axis=1)
            all_preds.extend(pred_labels)
            all_labels.extend(labels.numpy())
    
    return np.array(all_labels), np.array(all_preds)


def save_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    labels: list, 
    output_path: Path
) -> None:
    """Generate and save confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved to {output_path}")


def save_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    labels: list, 
    output_path: Path
) -> dict:
    """Generate classification report and save as JSON."""
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=labels, 
        output_dict=True,
        zero_division=0
    )
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Metrics saved to {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Train financial document classifier")
    parser.add_argument(
        "--data", 
        type=str, 
        required=True, 
        help="Path to CSV with 'text' and 'label' columns"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory to save model artifacts"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length")
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load and prepare data
    logger.info(f"Loading dataset from {args.data}")
    train_df, val_df = load_dataset(args.data)
    logger.info(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")

    # Encode labels
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_df["label"])
    val_labels = label_encoder.transform(val_df["label"])
    label_names = list(label_encoder.classes_)
    logger.info(f"Labels: {label_names}")

    # Initialize tokenizer and datasets
    logger.info("Loading tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    train_dataset = TextDataset(
        train_df["text"].tolist(), 
        train_labels, 
        tokenizer, 
        max_length=args.max_length
    )
    val_dataset = TextDataset(
        val_df["text"].tolist(), 
        val_labels, 
        tokenizer, 
        max_length=args.max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Create model
    logger.info("Initializing model...")
    model = create_model(num_labels=len(label_names))
    model.to(device)

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        logger.info(f"Epoch {epoch + 1}/{args.epochs} - Loss: {loss:.4f}")

    # Evaluation
    logger.info("Evaluating model on validation set...")
    y_true, y_pred = evaluate_model(model, val_loader, device)
    
    # Print classification report to console
    report_str = classification_report(y_true, y_pred, target_names=label_names, zero_division=0)
    logger.info(f"\nClassification Report:\n{report_str}")

    # Save artifacts
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save label encoder classes
    np.save(output_dir / "label_encoder.npy", label_encoder.classes_)
    
    # Save metrics and confusion matrix
    save_metrics(y_true, y_pred, label_names, output_dir / "metrics.json")
    save_confusion_matrix(y_true, y_pred, label_names, output_dir / "confusion_matrix.png")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
