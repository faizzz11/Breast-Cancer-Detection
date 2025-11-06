#!/usr/bin/env python3
"""
Training script for breast cancer detection models.

This script loads data, builds preprocessing pipelines, performs hyperparameter tuning,
and saves the best model for deployment.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    StratifiedKFold, RandomizedSearchCV, cross_validate
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline

# Import custom modules
from data_loader import load_data
from preprocessing import train_test_split_data, build_preprocessing_pipeline
from models import get_classifiers


def get_hyperparameter_grids() -> Dict[str, Dict[str, Any]]:
    """
    Get hyperparameter grids for model tuning.
    
    Returns:
        Dict[str, Dict[str, Any]]: Hyperparameter grids for each model
    """
    return {
        'random_forest': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__class_weight': ['balanced', None]
        },
        'logistic_regression': {
            'classifier__C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'classifier__penalty': ['l2'],
            'classifier__solver': ['liblinear', 'lbfgs'],
            'classifier__class_weight': ['balanced', None],
            'classifier__max_iter': [1000]
        }
    }


def create_full_pipeline(model_name: str, use_smote: bool = True) -> ImbPipeline:
    """
    Create a full pipeline with preprocessing and classifier.
    
    Args:
        model_name (str): Name of the model ('random_forest' or 'logistic_regression')
        use_smote (bool): Whether to use SMOTE for class balancing
    
    Returns:
        ImbPipeline: Complete pipeline with preprocessing and classifier
    """
    # Get feature names (will be set when we load data)
    # For now, create a placeholder - will be updated in main function
    
    if model_name == 'random_forest':
        classifier = RandomForestClassifier(random_state=42)
    elif model_name == 'logistic_regression':
        classifier = LogisticRegression(random_state=42)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create pipeline steps
    steps = [
        ('imputer', None),  # Will be set in main function
        ('scaler', None),   # Will be set in main function
    ]
    
    if use_smote:
        from imblearn.over_sampling import SMOTE
        steps.append(('smote', SMOTE(random_state=42)))
    
    steps.append(('classifier', classifier))
    
    return ImbPipeline(steps)


def evaluate_model_cv(pipeline: ImbPipeline, X: pd.DataFrame, y: pd.Series, 
                     cv: StratifiedKFold) -> Dict[str, float]:
    """
    Evaluate model using cross-validation.
    
    Args:
        pipeline: Trained pipeline
        X: Feature matrix
        y: Target vector
        cv: Cross-validation strategy
    
    Returns:
        Dict[str, float]: Cross-validation scores
    """
    scoring = ['roc_auc', 'precision', 'recall', 'f1']
    
    cv_results = cross_validate(
        pipeline, X, y, 
        cv=cv, 
        scoring=scoring, 
        n_jobs=-1,
        return_train_score=False
    )
    
    results = {}
    for metric in scoring:
        test_scores = cv_results[f'test_{metric}']
        results[f'{metric}_mean'] = np.mean(test_scores)
        results[f'{metric}_std'] = np.std(test_scores)
    
    return results


def train_and_evaluate_models(X_train: pd.DataFrame, y_train: pd.Series,
                             X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Train and evaluate multiple models with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
    
    Returns:
        Dict[str, Any]: Results including best model and metrics
    """
    print("Setting up cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Get hyperparameter grids
    param_grids = get_hyperparameter_grids()
    
    results = {}
    best_score = 0
    best_model = None
    best_model_name = None
    
    # Models to train
    models_to_train = ['random_forest', 'logistic_regression']
    
    for model_name in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training {model_name.replace('_', ' ').title()}")
        print(f"{'='*50}")
        
        # Create classifier
        if model_name == 'random_forest':
            classifier = RandomForestClassifier(random_state=42)
        else:  # logistic_regression
            classifier = LogisticRegression(random_state=42)
        
        # Create full pipeline with individual preprocessing steps
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from imblearn.over_sampling import SMOTE
        
        pipeline = ImbPipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('classifier', classifier)
        ])
        
        # Hyperparameter tuning
        print("Performing hyperparameter tuning...")
        param_grid = param_grids[model_name]
        
        # Use RandomizedSearchCV for efficiency
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=20,  # Number of parameter settings sampled
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        # Fit the search
        search.fit(X_train, y_train)
        
        print(f"Best parameters: {search.best_params_}")
        print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
        
        # Get the best pipeline
        best_pipeline = search.best_estimator_
        
        # Cross-validation evaluation
        print("Evaluating with cross-validation...")
        cv_results = evaluate_model_cv(best_pipeline, X_train, y_train, cv)
        
        print("Cross-validation results:")
        for metric, value in cv_results.items():
            if 'mean' in metric:
                std_key = metric.replace('mean', 'std')
                print(f"  {metric.replace('_mean', '').upper()}: {value:.4f} ± {cv_results[std_key]:.4f}")
        
        # Test set evaluation
        print("Evaluating on test set...")
        y_pred = best_pipeline.predict(X_test)
        y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print("Test set results:")
        for metric, value in test_metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        # Store results
        results[model_name] = {
            'best_params': search.best_params_,
            'best_cv_score': search.best_score_,
            'cv_results': cv_results,
            'test_metrics': test_metrics,
            'pipeline': best_pipeline
        }
        
        # Track best model
        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_model = best_pipeline
            best_model_name = model_name
    
    print(f"\n{'='*50}")
    print(f"Best model: {best_model_name.replace('_', ' ').title()}")
    print(f"Best CV ROC-AUC: {best_score:.4f}")
    print(f"{'='*50}")
    
    return {
        'results': results,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'best_score': best_score
    }


def save_model_and_results(model: ImbPipeline, results: Dict[str, Any], 
                          model_path: str = "models/final_model.joblib",
                          results_path: str = "outputs/training_results.json"):
    """
    Save the trained model and results.
    
    Args:
        model: Trained pipeline
        results: Training results
        model_path: Path to save the model
        results_path: Path to save the results
    """
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    # Save model
    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    
    # Prepare results for JSON serialization
    json_results = {}
    for model_name, model_results in results['results'].items():
        json_results[model_name] = {
            'best_params': model_results['best_params'],
            'best_cv_score': float(model_results['best_cv_score']),
            'cv_results': {k: float(v) for k, v in model_results['cv_results'].items()},
            'test_metrics': {k: float(v) for k, v in model_results['test_metrics'].items()}
        }
    
    json_results['best_model_name'] = results['best_model_name']
    json_results['best_score'] = float(results['best_score'])
    
    # Save results
    print(f"Saving results to {results_path}...")
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print("Model and results saved successfully!")


def main():
    """Main training function."""
    print("🔬 Breast Cancer Detection - Model Training")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    X, y = load_data(source="sklearn")
    print(f"Dataset shape: {X.shape}")
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train, y_train, X_test, y_test)
    
    # Save model and results
    save_model_and_results(results['best_model'], results)
    
    print("\n🎉 Training completed successfully!")
    print(f"Best model ({results['best_model_name']}) saved to models/final_model.joblib")
    print("Training results saved to outputs/training_results.json")


if __name__ == "__main__":
    main()