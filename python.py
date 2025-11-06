#!/usr/bin/env python3
"""
Simple demonstration script for the Breast Cancer Detection project.
This script shows basic usage of the data loading, preprocessing, and modeling modules.
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom modules
from data_loader import load_data, get_dataset_info
from preprocessing import get_preprocessing_pipeline, analyze_class_imbalance
from models import get_classifiers, get_classifier_info

def main():
    """Main demonstration function."""
    
    print("🔬 Breast Cancer Detection - Quick Demo")
    print("=" * 45)
    
    # Load and explore data
    print("\n📊 Loading and exploring data...")
    X, y = load_data(source="sklearn")
    
    # Dataset info
    info = get_dataset_info()
    print(f"Dataset: {info['n_samples']} samples, {info['n_features']} features")
    print(f"Classes: {info['class_distribution']}")
    
    # Class imbalance analysis
    imbalance = analyze_class_imbalance(y)
    print(f"Class imbalance ratio: {imbalance['imbalance_ratio']:.2f}")
    print(f"Minority class: {imbalance['minority_class']} ({imbalance['class_percentages'][imbalance['minority_class']]:.1f}%)")
    
    # Split data
    print("\n🔄 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Preprocessing
    print("\n⚙️ Preprocessing data...")
    pipeline = get_preprocessing_pipeline(scaler="standard", use_smote=True)
    X_train_processed, y_train_processed = pipeline.fit_resample(X_train, y_train)
    X_test_processed = pipeline.named_steps['scaler'].transform(X_test)
    
    print(f"After SMOTE: {X_train_processed.shape[0]} training samples")
    
    # Model training and evaluation
    print("\n🤖 Training and evaluating models...")
    models = get_classifiers(random_state=42)
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train_processed, y_train_processed)
        
        # Make predictions
        y_pred = model.predict(X_test_processed)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        print(f"  Accuracy: {accuracy:.3f}")
    
    # Results summary
    print("\n📈 Results Summary:")
    print("-" * 30)
    for name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:20}: {accuracy:.3f}")
    
    # Best model analysis
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    best_accuracy = results[best_model_name]
    
    print(f"\n🏆 Best model: {best_model_name} (Accuracy: {best_accuracy:.3f})")
    
    # Detailed evaluation of best model
    print(f"\n📊 Detailed evaluation of {best_model_name}:")
    y_pred_best = best_model.predict(X_test_processed)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_best, 
                              target_names=['Benign', 'Malignant']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_best)
    print("\nConfusion Matrix:")
    print(f"True Negatives (Benign correctly classified): {cm[0,0]}")
    print(f"False Positives (Benign misclassified as Malignant): {cm[0,1]}")
    print(f"False Negatives (Malignant misclassified as Benign): {cm[1,0]}")
    print(f"True Positives (Malignant correctly classified): {cm[1,1]}")
    
    # Feature importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        print(f"\n🔍 Top 5 most important features for {best_model_name}:")
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, (_, row) in enumerate(feature_importance.head().iterrows()):
            print(f"  {i+1}. {row['feature']}: {row['importance']:.3f}")
    
    elif hasattr(best_model, 'coef_'):
        print(f"\n🔍 Top 5 most important features for {best_model_name}:")
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(best_model.coef_[0])
        }).sort_values('importance', ascending=False)
        
        for i, (_, row) in enumerate(feature_importance.head().iterrows()):
            print(f"  {i+1}. {row['feature']}: {row['importance']:.3f}")
    
    print("\n✨ Demo completed successfully!")
    print("\nFor more detailed analysis, check out the Jupyter notebooks:")
    print("  • notebooks/01_eda.ipynb - Exploratory Data Analysis")
    print("  • notebooks/02_preprocessing.ipynb - Data Preprocessing")
    print("  • notebooks/03_modeling.ipynb - Advanced Modeling")

if __name__ == "__main__":
    main()