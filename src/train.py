"""
Training script for the financial document classifier.

Supports:
- Standard train/validation split
- K-fold cross-validation for robust evaluation
- Class weight balancing for imbalanced datasets
- Stratified splitting to maintain class distribution

Usage:
    # Standard training
    python -m src.train --data data/sample_dataset.csv --output_dir models/run1
    
    # With cross-validation
    python -m src.train --data data/sample_dataset.csv --output_dir models/run1 --cross_validate --n_folds 5
"""

import argparse
import json
import logging
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

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


def compute_class_weights(labels: np.ndarray, device) -> torch.Tensor:
    """
    Compute class weights to handle imbalanced datasets.
    
    Uses sklearn's compute_class_weight with 'balanced' strategy:
    weight_i = n_samples / (n_classes * n_samples_i)
    
    Args:
        labels: Array of label indices
        device: PyTorch device
        
    Returns:
        Tensor of class weights
    """
    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=labels
    )
    logger.info(f"Class weights computed: {dict(zip(classes, weights.round(3)))}")
    return torch.tensor(weights, dtype=torch.float).to(device)


def train_epoch(
    model, 
    dataloader: DataLoader, 
    optimizer, 
    scheduler, 
    device,
    class_weights: torch.Tensor = None
) -> float:
    """Run one training epoch and return average loss."""
    model.train()
    total_loss = 0.0
    
    # Use weighted cross-entropy if class weights provided
    criterion = None
    if class_weights is not None:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        if criterion is not None:
            # Manual loss computation with class weights
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            loss = criterion(outputs.logits, batch["labels"])
        else:
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
    output_path: Path,
    extra_info: dict = None
) -> dict:
    """Generate classification report and save as JSON."""
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=labels, 
        output_dict=True,
        zero_division=0
    )
    
    # Add extra info if provided
    if extra_info:
        report["training_info"] = extra_info
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Metrics saved to {output_path}")
    return report


def train_single_fold(
    train_texts, train_labels,
    val_texts, val_labels,
    label_encoder,
    tokenizer,
    args,
    device,
    class_weights=None
):
    """Train model on a single fold and return evaluation results."""
    
    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Create model
    model = create_model(num_labels=len(label_encoder.classes_))
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
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, scheduler, device, class_weights)
        logger.info(f"Epoch {epoch + 1}/{args.epochs} - Loss: {loss:.4f}")
    
    # Evaluation
    y_true, y_pred = evaluate_model(model, val_loader, device)
    
    return model, y_true, y_pred


def run_cross_validation(df, label_encoder, tokenizer, args, device):
    """
    Run k-fold stratified cross-validation.
    
    Returns aggregated results across all folds.
    """
    logger.info(f"Running {args.n_folds}-fold stratified cross-validation...")
    
    texts = df["text"].tolist()
    labels = label_encoder.transform(df["label"])
    
    # Compute class weights once
    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(labels, device)
    
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    
    all_y_true = []
    all_y_pred = []
    fold_accuracies = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        logger.info(f"\n{'='*50}")
        logger.info(f"Fold {fold + 1}/{args.n_folds}")
        logger.info(f"{'='*50}")
        
        train_texts = [texts[i] for i in train_idx]
        train_labels = labels[train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = labels[val_idx]
        
        logger.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}")
        
        model, y_true, y_pred = train_single_fold(
            train_texts, train_labels,
            val_texts, val_labels,
            label_encoder, tokenizer, args, device, class_weights
        )
        
        fold_acc = (y_true == y_pred).mean()
        fold_accuracies.append(fold_acc)
        logger.info(f"Fold {fold + 1} Accuracy: {fold_acc:.4f}")
        
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
    
    # Aggregate results
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Cross-Validation Results ({args.n_folds} folds)")
    logger.info(f"{'='*50}")
    logger.info(f"Mean Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")
    logger.info(f"Per-fold accuracies: {[f'{a:.4f}' for a in fold_accuracies]}")
    
    return all_y_true, all_y_pred, {
        "n_folds": args.n_folds,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "fold_accuracies": fold_accuracies
    }


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
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Data split arguments
    parser.add_argument(
        "--test_size", 
        type=float, 
        default=0.2, 
        help="Fraction of data for validation/test (default: 0.2)"
    )
    parser.add_argument(
        "--holdout_test", 
        action="store_true",
        help="Use 3-way split: train/val/test instead of train/val"
    )
    
    # Cross-validation arguments
    parser.add_argument(
        "--cross_validate", 
        action="store_true",
        help="Run k-fold cross-validation instead of single split"
    )
    parser.add_argument(
        "--n_folds", 
        type=int, 
        default=5, 
        help="Number of folds for cross-validation"
    )
    
    # Class imbalance handling
    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        help="Use class weights to handle imbalanced datasets"
    )
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Random seed: {args.seed}")

    # Load data
    logger.info(f"Loading dataset from {args.data}")
    df = pd.read_csv(args.data)
    logger.info(f"Total samples: {len(df)}")
    
    # Show class distribution
    class_dist = df["label"].value_counts()
    logger.info(f"Class distribution:\n{class_dist.to_string()}")
    
    # Check for class imbalance
    imbalance_ratio = class_dist.max() / class_dist.min()
    if imbalance_ratio > 1.5:
        logger.warning(f"Class imbalance detected (ratio: {imbalance_ratio:.2f})")
        if not args.use_class_weights:
            logger.warning("Consider using --use_class_weights flag")

    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(df["label"])
    label_names = list(label_encoder.classes_)
    logger.info(f"Labels: {label_names}")

    # Initialize tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # Training info for logging
    training_info = {
        "dataset_size": len(df),
        "num_classes": len(label_names),
        "class_distribution": class_dist.to_dict(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_length": args.max_length,
        "seed": args.seed,
        "use_class_weights": args.use_class_weights
    }

    if args.cross_validate:
        # Cross-validation mode
        training_info["evaluation_strategy"] = f"{args.n_folds}-fold stratified cross-validation"
        
        y_true, y_pred, cv_results = run_cross_validation(
            df, label_encoder, tokenizer, args, device
        )
        training_info["cross_validation"] = cv_results
        
        # Train final model on all data for deployment
        logger.info("\nTraining final model on full dataset...")
        all_labels = label_encoder.transform(df["label"])
        class_weights = None
        if args.use_class_weights:
            class_weights = compute_class_weights(all_labels, device)
        
        final_model, _, _ = train_single_fold(
            df["text"].tolist(), all_labels,
            df["text"].tolist()[:10], all_labels[:10],  # Dummy val set
            label_encoder, tokenizer, args, device, class_weights
        )
        model = final_model
        
    else:
        # Standard split mode
        if args.holdout_test:
            # 3-way split: train/val/test
            training_info["evaluation_strategy"] = "train/val/test split (60/20/20)"
            
            # First split: separate test set
            train_val_df, test_df = train_test_split(
                df, test_size=args.test_size, random_state=args.seed, stratify=df["label"]
            )
            # Second split: separate validation set
            train_df, val_df = train_test_split(
                train_val_df, test_size=0.25, random_state=args.seed, stratify=train_val_df["label"]
            )
            
            logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            training_info["split_sizes"] = {
                "train": len(train_df),
                "validation": len(val_df),
                "test": len(test_df)
            }
            
            # Train on train set, validate on val set
            train_labels = label_encoder.transform(train_df["label"])
            val_labels = label_encoder.transform(val_df["label"])
            test_labels = label_encoder.transform(test_df["label"])
            
            class_weights = None
            if args.use_class_weights:
                class_weights = compute_class_weights(train_labels, device)
            
            model, _, _ = train_single_fold(
                train_df["text"].tolist(), train_labels,
                val_df["text"].tolist(), val_labels,
                label_encoder, tokenizer, args, device, class_weights
            )
            
            # Final evaluation on held-out test set
            logger.info("\nEvaluating on held-out test set...")
            test_dataset = TextDataset(test_df["text"].tolist(), test_labels, tokenizer, args.max_length)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
            y_true, y_pred = evaluate_model(model, test_loader, device)
            
        else:
            # 2-way split: train/val
            training_info["evaluation_strategy"] = f"stratified train/val split ({int((1-args.test_size)*100)}/{int(args.test_size*100)})"
            
            train_df, val_df = train_test_split(
                df, test_size=args.test_size, random_state=args.seed, stratify=df["label"]
            )
            
            logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")
            training_info["split_sizes"] = {
                "train": len(train_df),
                "validation": len(val_df)
            }
            
            train_labels = label_encoder.transform(train_df["label"])
            val_labels = label_encoder.transform(val_df["label"])
            
            class_weights = None
            if args.use_class_weights:
                class_weights = compute_class_weights(train_labels, device)
            
            model, y_true, y_pred = train_single_fold(
                train_df["text"].tolist(), train_labels,
                val_df["text"].tolist(), val_labels,
                label_encoder, tokenizer, args, device, class_weights
            )

    # Print classification report
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
    save_metrics(y_true, y_pred, label_names, output_dir / "metrics.json", training_info)
    save_confusion_matrix(y_true, y_pred, label_names, output_dir / "confusion_matrix.png")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
