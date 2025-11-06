#!/usr/bin/env python3
"""
Model explainability script for breast cancer detection.

This script provides model interpretability through feature importance analysis
and SHAP (SHapley Additive exPlanations) values.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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


def extract_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract feature importance from tree-based models.
    
    Args:
        model: Trained model pipeline
        feature_names: List of feature names
    
    Returns:
        pd.DataFrame: Feature importance dataframe
    """
    # Get the classifier from the pipeline
    classifier = model.named_steps.get('classifier', model)
    
    if hasattr(classifier, 'feature_importances_'):
        # Tree-based models
        importances = classifier.feature_importances_
        importance_type = "Tree-based Feature Importance"
    elif hasattr(classifier, 'coef_'):
        # Linear models
        importances = np.abs(classifier.coef_[0])
        importance_type = "Linear Model Coefficient Magnitude"
    else:
        print("Model does not support feature importance extraction")
        return pd.DataFrame()
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    importance_df['importance_normalized'] = importance_df['importance'] / importance_df['importance'].sum()
    
    print(f"Extracted {importance_type}")
    return importance_df


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15,
                          save_path: str = "outputs/feature_importance.png"):
    """
    Plot feature importance.
    
    Args:
        importance_df: Feature importance dataframe
        top_n: Number of top features to plot
        save_path: Path to save the plot
    """
    if importance_df.empty:
        print("No feature importance data to plot")
        return
    
    plt.figure(figsize=(10, 8))
    
    # Get top N features
    top_features = importance_df.head(top_n)
    
    # Create horizontal bar plot
    bars = plt.barh(range(len(top_features)), top_features['importance'], 
                    color='skyblue', alpha=0.7, edgecolor='black')
    
    # Customize plot
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # Highest importance at top
    
    # Add value labels
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{importance:.3f}', ha='left', va='center', fontsize=9)
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to {save_path}")


def compute_shap_values(model, X_sample: pd.DataFrame, 
                       background_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Compute SHAP values for model explanations.
    
    Args:
        model: Trained model pipeline
        X_sample: Sample data for SHAP analysis
        background_data: Background data for SHAP explainer
    
    Returns:
        Dict[str, Any]: SHAP values and explainer
    """
    if not SHAP_AVAILABLE:
        print("SHAP not available. Skipping SHAP analysis.")
        return {}
    
    print("Computing SHAP values...")
    
    # Get the classifier from the pipeline
    classifier = model.named_steps.get('classifier', model)
    
    # Transform the data through preprocessing steps
    X_transformed = X_sample.copy()
    for step_name, transformer in model.named_steps.items():
        if step_name != 'classifier' and step_name != 'smote':
            if hasattr(transformer, 'transform'):
                X_transformed = transformer.transform(X_transformed)
    
    # Convert to numpy array if needed
    if isinstance(X_transformed, pd.DataFrame):
        X_transformed = X_transformed.values
    
    # Choose appropriate explainer based on model type
    if isinstance(classifier, (RandomForestClassifier,)):
        # Tree explainer for tree-based models
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_transformed)
        
        # For binary classification, get positive class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class (malignant)
            
    elif isinstance(classifier, (LogisticRegression,)):
        # Linear explainer for linear models
        explainer = shap.LinearExplainer(classifier, X_transformed)
        shap_values = explainer.shap_values(X_transformed)
        
    else:
        # Kernel explainer as fallback (slower but works for any model)
        if background_data is not None:
            # Transform background data
            bg_transformed = background_data.copy()
            for step_name, transformer in model.named_steps.items():
                if step_name != 'classifier' and step_name != 'smote':
                    if hasattr(transformer, 'transform'):
                        bg_transformed = transformer.transform(bg_transformed)
            if isinstance(bg_transformed, pd.DataFrame):
                bg_transformed = bg_transformed.values
        else:
            bg_transformed = X_transformed[:100]  # Use subset as background
        
        explainer = shap.KernelExplainer(classifier.predict_proba, bg_transformed)
        shap_values = explainer.shap_values(X_transformed)
        
        # For binary classification, get positive class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class (malignant)
    
    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'X_transformed': X_transformed
    }


def plot_shap_summary(shap_data: Dict[str, Any], feature_names: List[str],
                     save_path: str = "outputs/shap/shap_summary.png"):
    """
    Create SHAP summary plot.
    
    Args:
        shap_data: SHAP data dictionary
        feature_names: List of feature names
        save_path: Path to save the plot
    """
    if not shap_data or not SHAP_AVAILABLE:
        print("SHAP data not available. Skipping summary plot.")
        return
    
    # Create output directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # Create SHAP summary plot
    shap.summary_plot(
        shap_data['shap_values'], 
        shap_data['X_transformed'],
        feature_names=feature_names,
        show=False,
        max_display=15
    )
    
    plt.title('SHAP Summary Plot', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP summary plot saved to {save_path}")


def plot_shap_force_plot(shap_data: Dict[str, Any], feature_names: List[str],
                        instance_idx: int = 0, 
                        save_path: str = "outputs/shap/shap_force_plot.png"):
    """
    Create SHAP force plot for a single instance.
    
    Args:
        shap_data: SHAP data dictionary
        feature_names: List of feature names
        instance_idx: Index of instance to explain
        save_path: Path to save the plot
    """
    if not shap_data or not SHAP_AVAILABLE:
        print("SHAP data not available. Skipping force plot.")
        return
    
    # Create output directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        # Create force plot
        force_plot = shap.force_plot(
            shap_data['explainer'].expected_value,
            shap_data['shap_values'][instance_idx],
            shap_data['X_transformed'][instance_idx],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        
        plt.title(f'SHAP Force Plot - Instance {instance_idx}', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"SHAP force plot saved to {save_path}")
        
    except Exception as e:
        print(f"Could not create force plot: {e}")
        
        # Create alternative waterfall plot
        try:
            plt.figure(figsize=(10, 8))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_data['shap_values'][instance_idx],
                    base_values=shap_data['explainer'].expected_value,
                    data=shap_data['X_transformed'][instance_idx],
                    feature_names=feature_names
                ),
                show=False
            )
            plt.title(f'SHAP Waterfall Plot - Instance {instance_idx}', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_path.replace('force_plot', 'waterfall_plot'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"SHAP waterfall plot saved to {save_path.replace('force_plot', 'waterfall_plot')}")
        except Exception as e2:
            print(f"Could not create waterfall plot either: {e2}")


def analyze_model_explainability(model_path: str = "models/final_model.joblib",
                               n_samples: int = 100) -> Dict[str, Any]:
    """
    Perform comprehensive model explainability analysis.
    
    Args:
        model_path: Path to the trained model
        n_samples: Number of samples for SHAP analysis
    
    Returns:
        Dict[str, Any]: Explainability results
    """
    # Load model
    model = load_model(model_path)
    
    # Load data
    print("Loading data...")
    X, y = load_data(source="sklearn")
    _, X_test, _, y_test = train_test_split_data(X, y, test_size=0.2, random_state=42)
    
    feature_names = X.columns.tolist()
    
    # Extract feature importance
    print("Extracting feature importance...")
    importance_df = extract_feature_importance(model, feature_names)
    
    if not importance_df.empty:
        # Plot feature importance
        plot_feature_importance(importance_df)
        
        # Print top features
        print("\nTop 10 Most Important Features:")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
            print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
    
    # SHAP analysis
    if SHAP_AVAILABLE:
        print(f"\nPerforming SHAP analysis on {n_samples} samples...")
        
        # Use a subset of test data for SHAP analysis
        X_sample = X_test.head(n_samples)
        background_data = X_test.head(50)  # Smaller background for efficiency
        
        # Compute SHAP values
        shap_data = compute_shap_values(model, X_sample, background_data)
        
        if shap_data:
            # Create SHAP plots
            plot_shap_summary(shap_data, feature_names)
            plot_shap_force_plot(shap_data, feature_names, instance_idx=0)
            
            # Try a few more instances
            for idx in [1, 2]:
                if idx < len(X_sample):
                    plot_shap_force_plot(
                        shap_data, feature_names, instance_idx=idx,
                        save_path=f"outputs/shap/shap_force_plot_instance_{idx}.png"
                    )
    
    return {
        'feature_importance': importance_df,
        'shap_data': shap_data if SHAP_AVAILABLE else None
    }


def main():
    """Main explainability function."""
    print("🔬 Breast Cancer Detection - Model Explainability")
    print("=" * 50)
    
    # Check if model exists
    model_path = "models/final_model.joblib"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please run 'python src/train.py' first to train a model.")
        return
    
    # Analyze explainability
    results = analyze_model_explainability(model_path)
    
    print("\n🎉 Explainability analysis completed successfully!")
    print("📊 Feature importance plot saved to outputs/feature_importance.png")
    if SHAP_AVAILABLE:
        print("🔍 SHAP plots saved to outputs/shap/")
    else:
        print("⚠️  SHAP analysis skipped (install shap package for SHAP analysis)")


if __name__ == "__main__":
    main()