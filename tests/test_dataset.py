"""
Tests for the dataset loading utilities.

Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
from pathlib import Path

from src.dataset import load_dataset


DATA_PATH = Path(__file__).parent.parent / "data" / "sample_dataset.csv"


class TestLoadDataset:
    """Tests for the load_dataset function."""
    
    def test_load_dataset_returns_dataframes(self):
        """load_dataset should return two DataFrames."""
        train_df, val_df = load_dataset(DATA_PATH)
        
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(val_df, pd.DataFrame)
    
    def test_load_dataset_has_required_columns(self):
        """Both DataFrames should have 'text' and 'label' columns."""
        train_df, val_df = load_dataset(DATA_PATH)
        
        assert "text" in train_df.columns
        assert "label" in train_df.columns
        assert "text" in val_df.columns
        assert "label" in val_df.columns
    
    def test_load_dataset_split_sizes(self):
        """Train set should be larger than validation set with default split."""
        train_df, val_df = load_dataset(DATA_PATH)
        
        assert len(train_df) > len(val_df)
        # Default is 80/20 split
        total = len(train_df) + len(val_df)
        assert len(train_df) / total >= 0.75
    
    def test_load_dataset_custom_split(self):
        """load_dataset should respect custom test_size."""
        train_df, val_df = load_dataset(DATA_PATH, test_size=0.3)
        
        total = len(train_df) + len(val_df)
        val_ratio = len(val_df) / total
        # Should be approximately 30% validation
        assert 0.25 <= val_ratio <= 0.35
    
    def test_load_dataset_stratified_split(self):
        """Split should be stratified by label."""
        train_df, val_df = load_dataset(DATA_PATH)
        
        # Both sets should contain samples from multiple classes
        train_labels = set(train_df["label"].unique())
        val_labels = set(val_df["label"].unique())
        
        # Should have overlap in labels (stratified)
        assert len(train_labels) >= 2
        assert len(val_labels) >= 2

