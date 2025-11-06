#!/usr/bin/env python3
"""
Model evaluation script for breast cancer detection.

This script loads a trained model and evaluates it on test data,
generating comprehensive metrics and visualizations.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score
)

# Import custom modules
from data_loader import load_data
from preprocessing import train_test_split_data


def load_model(model_path: str = "models/final_model.joblib"):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to the saved model
    
    Returns:
        Trained pipeline
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"Loading model from {model_path}...")
    return joblib.load(model_path)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                   y_pred_proba: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
    
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'pr_auc': average_precision_score(y_true, y_pred_proba)
    }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         save_path: str = "outputs/confusion_matrix.png"):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add percentage annotations
    total = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.7, f'({cm[i, j]/total:.1%})', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                   save_path: str = "outputs/roc_curve.png"):
    """
    Plot and save ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save the plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to {save_path}")


def plot_precision_recall_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                               save_path: str = "outputs/precision_recall_curve.png"):
    """
    Plot and save Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save the plot
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR Curve (AUC = {pr_auc:.3f})')
    
    # Baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)
    plt.axhline(y=baseline, color='red', linestyle='--', lw=2,
                label=f'Random Classifier (AP = {baseline:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Precision-Recall curve saved to {save_path}")


def save_metrics_json(metrics: Dict[str, float], classification_rep: str,
                     save_path: str = "outputs/metrics.json"):
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        classification_rep: Classification report string
        save_path: Path to save the JSON file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    output = {
        'metrics': {k: float(v) for k, v in metrics.items()},
        'classification_report': classification_rep
    }
    
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Metrics saved to {save_path}")


def evaluate_model(model_path: str = "models/final_model.joblib",
                  X_test: Optional[pd.DataFrame] = None,
                  y_test: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    Evaluate a trained model and generate comprehensive results.
    
    Args:
        model_path: Path to the trained model
        X_test: Test features (if None, will load and split data)
        y_test: Test targets (if None, will load and split data)
    
    Returns:
        Dict[str, Any]: Evaluation results
    """
    # Load model
    model = load_model(model_path)
    
    # Load test data if not provided
    if X_test is None or y_test is None:
        print("Loading and splitting data...")
        X, y = load_data(source="sklearn")
        _, X_test, _, y_test = train_test_split_data(X, y, test_size=0.2, random_state=42)
    
    print(f"Evaluating on {len(X_test)} test samples...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    metrics = compute_metrics(y_test, y_pred, y_pred_proba)
    
    # Generate classification report
    class_report = classification_report(y_test, y_pred, 
                                       target_names=['Benign', 'Malignant'])
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"  {metric.upper().replace('_', ' ')}: {value:.4f}")
    
    print(f"\nClassification Report:")
    print(class_report)
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  True Negatives (Benign correctly classified): {cm[0,0]}")
    print(f"  False Positives (Benign misclassified as Malignant): {cm[0,1]}")
    print(f"  False Negatives (Malignant misclassified as Benign): {cm[1,0]}")
    print(f"  True Positives (Malignant correctly classified): {cm[1,1]}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_pred_proba)
    plot_precision_recall_curve(y_test, y_pred_proba)
    
    # Save metrics
    save_metrics_json(metrics, class_report)
    
    return {
        'metrics': metrics,
        'classification_report': class_report,
        'confusion_matrix': cm,
        'y_true': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


def main():
    """Main evaluation function."""
    print("🔬 Breast Cancer Detection - Model Evaluation")
    print("=" * 50)
    
    # Check if model exists
    model_path = "models/final_model.joblib"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please run 'python src/train.py' first to train a model.")
        return
    
    # Evaluate model
    results = evaluate_model(model_path)
    
    print("\n🎉 Evaluation completed successfully!")
    print("📊 Visualizations saved to outputs/")
    print("📋 Metrics saved to outputs/metrics.json")


if __name__ == "__main__":
    main()