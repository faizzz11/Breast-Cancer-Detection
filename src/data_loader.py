"""
Data loading utilities for the Wisconsin Breast Cancer dataset.

This module provides functions to load the breast cancer dataset from different sources:
- sklearn.datasets.load_breast_cancer (default)
- CSV file from data/raw/wdbc.data
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.datasets import load_breast_cancer
import os


def load_data(source: str = "sklearn", path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the Wisconsin Breast Cancer dataset from different sources.
    
    Args:
        source (str): Data source - either "sklearn" or "csv"
        path (str, optional): Path to CSV file when source="csv". 
                              Defaults to "data/raw/wdbc.data"
    
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features (X) and target (y)
            - X: DataFrame with 30 feature columns
            - y: Series with binary target (0=benign, 1=malignant)
    
    Raises:
        ValueError: If source is not "sklearn" or "csv"
        FileNotFoundError: If CSV file doesn't exist when source="csv"
    """
    if source == "sklearn":
        return _load_from_sklearn()
    elif source == "csv":
        csv_path = path or "data/raw/wdbc.data"
        return _load_from_csv(csv_path)
    else:
        raise ValueError(f"Source must be 'sklearn' or 'csv', got '{source}'")


def _load_from_sklearn() -> Tuple[pd.DataFrame, pd.Series]:
    """Load data from sklearn.datasets.load_breast_cancer."""
    data = load_breast_cancer()
    
    # Create DataFrame with feature names
    X = pd.DataFrame(data.data, columns=data.feature_names)
    
    # Create Series for target (0=benign, 1=malignant)
    y = pd.Series(data.target, name='diagnosis')
    
    return X, y


def _load_from_csv(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load data from CSV file (wdbc.data format).
    
    The CSV format is:
    - Column 0: ID (ignored)
    - Column 1: Diagnosis (M=malignant, B=benign)
    - Columns 2-31: 30 feature values
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Define feature names based on WDBC dataset documentation
    feature_names = _get_feature_names()
    
    # Read CSV without header
    df = pd.read_csv(csv_path, header=None)
    
    # Extract features (columns 2-31)
    X = df.iloc[:, 2:32].copy()
    X.columns = feature_names
    
    # Extract and encode target (column 1: M=1, B=0)
    y = df.iloc[:, 1].map({'M': 1, 'B': 0})
    y.name = 'diagnosis'
    
    return X, y


def _get_feature_names() -> list:
    """
    Get the 30 feature names for the Wisconsin Breast Cancer dataset.
    
    Features are computed for each cell nucleus:
    - 10 base features (radius, texture, perimeter, area, smoothness, 
      compactness, concavity, concave_points, symmetry, fractal_dimension)
    - Each computed as: mean, standard error (se), and worst (largest)
    - Total: 10 × 3 = 30 features
    """
    base_features = [
        'radius', 'texture', 'perimeter', 'area', 'smoothness',
        'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension'
    ]
    
    feature_names = []
    
    # Mean features (0-9)
    for feature in base_features:
        feature_names.append(f'mean_{feature}')
    
    # Standard error features (10-19)
    for feature in base_features:
        feature_names.append(f'se_{feature}')
    
    # Worst features (20-29)
    for feature in base_features:
        feature_names.append(f'worst_{feature}')
    
    return feature_names


def get_dataset_info() -> dict:
    """
    Get information about the Wisconsin Breast Cancer dataset.
    
    Returns:
        dict: Dataset information including shape, class distribution, etc.
    """
    X, y = load_data(source="sklearn")
    
    info = {
        'n_samples': len(X),
        'n_features': len(X.columns),
        'feature_names': list(X.columns),
        'class_distribution': {
            'benign (0)': (y == 0).sum(),
            'malignant (1)': (y == 1).sum()
        },
        'class_balance_ratio': (y == 0).sum() / (y == 1).sum(),
        'missing_values': X.isnull().sum().sum()
    }
    
    return info


if __name__ == "__main__":
    # Example usage
    print("Loading data from sklearn...")
    X_sklearn, y_sklearn = load_data(source="sklearn")
    print(f"Sklearn data shape: X={X_sklearn.shape}, y={y_sklearn.shape}")
    
    print("\nLoading data from CSV...")
    try:
        X_csv, y_csv = load_data(source="csv")
        print(f"CSV data shape: X={X_csv.shape}, y={y_csv.shape}")
        
        # Verify data consistency
        print(f"\nData consistency check:")
        print(f"Features match: {X_sklearn.shape == X_csv.shape}")
        print(f"Targets match: {y_sklearn.shape == y_csv.shape}")
        
    except FileNotFoundError as e:
        print(f"CSV file not found: {e}")
    
    print("\nDataset info:")
    info = get_dataset_info()
    for key, value in info.items():
        print(f"{key}: {value}")