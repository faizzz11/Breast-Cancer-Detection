"""
Machine learning models for breast cancer detection.

This module provides functions to get pre-configured classifiers with
sensible default parameters for the breast cancer classification task.
"""

import numpy as np
from typing import Dict, Any, List
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator


def get_classifiers(random_state: int = 42) -> Dict[str, BaseEstimator]:
    """
    Get a dictionary of pre-configured classifiers for breast cancer detection.
    
    All models are configured with sensible defaults and the same random_state
    for reproducibility.
    
    Args:
        random_state (int): Random state for reproducibility
    
    Returns:
        Dict[str, BaseEstimator]: Dictionary mapping model names to fitted models
            - 'logistic_regression': LogisticRegression
            - 'random_forest': RandomForestClassifier  
            - 'svm': SVC with probability=True
            - 'xgboost': XGBClassifier
    """
    classifiers = {
        'logistic_regression': LogisticRegression(
            random_state=random_state,
            max_iter=1000,  # Increased for convergence
            solver='liblinear',  # Good for small datasets
            C=1.0,  # Regularization strength
            penalty='l2'  # L2 regularization
        ),
        
        'random_forest': RandomForestClassifier(
            random_state=random_state,
            n_estimators=100,  # Number of trees
            max_depth=None,  # No limit on depth
            min_samples_split=2,  # Min samples to split node
            min_samples_leaf=1,  # Min samples in leaf
            max_features='sqrt',  # Features to consider for best split
            bootstrap=True,  # Bootstrap sampling
            oob_score=True  # Out-of-bag score
        ),
        
        'svm': SVC(
            random_state=random_state,
            probability=True,  # Enable probability estimates
            kernel='rbf',  # Radial basis function kernel
            C=1.0,  # Regularization parameter
            gamma='scale',  # Kernel coefficient
            class_weight='balanced'  # Handle class imbalance
        ),
        
        'xgboost': XGBClassifier(
            random_state=random_state,
            n_estimators=100,  # Number of boosting rounds
            max_depth=6,  # Maximum tree depth
            learning_rate=0.1,  # Step size shrinkage
            subsample=0.8,  # Subsample ratio of training instances
            colsample_bytree=0.8,  # Subsample ratio of features
            reg_alpha=0,  # L1 regularization
            reg_lambda=1,  # L2 regularization
            eval_metric='logloss',  # Evaluation metric
            use_label_encoder=False  # Avoid deprecation warning
        )
    }
    
    return classifiers


def get_classifier_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about each classifier including strengths and use cases.
    
    Returns:
        Dict[str, Dict[str, Any]]: Information about each classifier
    """
    info = {
        'logistic_regression': {
            'name': 'Logistic Regression',
            'type': 'Linear',
            'strengths': [
                'Fast training and prediction',
                'Interpretable coefficients',
                'Good baseline model',
                'Probabilistic output',
                'No hyperparameter tuning needed'
            ],
            'weaknesses': [
                'Assumes linear relationship',
                'Sensitive to outliers',
                'May underfit complex patterns'
            ],
            'best_for': 'Baseline model, interpretability, linear relationships'
        },
        
        'random_forest': {
            'name': 'Random Forest',
            'type': 'Ensemble (Bagging)',
            'strengths': [
                'Handles non-linear relationships',
                'Feature importance scores',
                'Robust to outliers',
                'Less prone to overfitting',
                'Works well out-of-the-box'
            ],
            'weaknesses': [
                'Can overfit with very noisy data',
                'Less interpretable than single trees',
                'Memory intensive'
            ],
            'best_for': 'General-purpose, feature importance, robust predictions'
        },
        
        'svm': {
            'name': 'Support Vector Machine',
            'type': 'Kernel-based',
            'strengths': [
                'Effective in high dimensions',
                'Memory efficient',
                'Versatile (different kernels)',
                'Works well with small datasets'
            ],
            'weaknesses': [
                'Slow on large datasets',
                'Sensitive to feature scaling',
                'No probabilistic output (without probability=True)',
                'Hyperparameter sensitive'
            ],
            'best_for': 'High-dimensional data, small datasets, non-linear patterns'
        },
        
        'xgboost': {
            'name': 'XGBoost',
            'type': 'Ensemble (Boosting)',
            'strengths': [
                'Often achieves best performance',
                'Handles missing values',
                'Built-in regularization',
                'Feature importance scores',
                'Efficient implementation'
            ],
            'weaknesses': [
                'Many hyperparameters to tune',
                'Can overfit easily',
                'Less interpretable',
                'Requires more computational resources'
            ],
            'best_for': 'Maximum performance, competitions, complex patterns'
        }
    }
    
    return info


def get_hyperparameter_grids(random_state: int = 42) -> Dict[str, Dict[str, List]]:
    """
    Get hyperparameter grids for GridSearchCV or RandomizedSearchCV.
    
    Args:
        random_state (int): Random state for reproducibility
    
    Returns:
        Dict[str, Dict[str, List]]: Hyperparameter grids for each model
    """
    grids = {
        'logistic_regression': {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000]
        },
        
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        },
        
        'svm': {
            'C': [0.1, 1.0, 10.0, 100.0],
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
            'class_weight': ['balanced', None]
        },
        
        'xgboost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1.0],
            'reg_lambda': [1, 1.5, 2.0]
        }
    }
    
    return grids


def get_quick_hyperparameter_grids(random_state: int = 42) -> Dict[str, Dict[str, List]]:
    """
    Get smaller hyperparameter grids for quick experimentation.
    
    Args:
        random_state (int): Random state for reproducibility
    
    Returns:
        Dict[str, Dict[str, List]]: Smaller hyperparameter grids
    """
    quick_grids = {
        'logistic_regression': {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l2']
        },
        
        'random_forest': {
            'n_estimators': [50, 100],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        },
        
        'svm': {
            'C': [1.0, 10.0],
            'kernel': ['rbf'],
            'gamma': ['scale', 0.1]
        },
        
        'xgboost': {
            'n_estimators': [50, 100],
            'max_depth': [3, 6],
            'learning_rate': [0.1, 0.2]
        }
    }
    
    return quick_grids


if __name__ == "__main__":
    # Example usage
    print("Getting classifiers...")
    classifiers = get_classifiers(random_state=42)
    
    print(f"\nAvailable classifiers: {list(classifiers.keys())}")
    
    print("\nClassifier information:")
    classifier_info = get_classifier_info()
    for name, info in classifier_info.items():
        print(f"\n{info['name']} ({info['type']}):")
        print(f"  Best for: {info['best_for']}")
        print(f"  Strengths: {', '.join(info['strengths'][:3])}...")
    
    print("\nHyperparameter grids available:")
    grids = get_hyperparameter_grids()
    for name, grid in grids.items():
        print(f"  {name}: {len(grid)} parameters")
    
    print("\nQuick grids available:")
    quick_grids = get_quick_hyperparameter_grids()
    for name, grid in quick_grids.items():
        print(f"  {name}: {len(grid)} parameters")
    
    # Test instantiation
    print("\nTesting model instantiation...")
    for name, model in classifiers.items():
        print(f"  {name}: {type(model).__name__} - OK")