#!/usr/bin/env python3
"""
Example usage script demonstrating the complete breast cancer detection workflow.

This script shows how to use all the components of the project:
1. Training models
2. Evaluating models
3. Generating explanations
4. Making predictions
"""

import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.append('src')

from data_loader import load_data
from preprocessing import train_test_split_data
from train import main as train_main
from evaluate import main as evaluate_main
from explainability import main as explainability_main

def demonstrate_workflow():
    """Demonstrate the complete workflow."""
    
    print("🔬 Breast Cancer Detection - Complete Workflow Demo")
    print("=" * 60)
    
    # Step 1: Load and explore data
    print("\n1️⃣ Loading and exploring data...")
    X, y = load_data(source="sklearn")
    print(f"   Dataset shape: {X.shape}")
    print(f"   Features: {list(X.columns[:5])}... (showing first 5)")
    print(f"   Class distribution: {y.value_counts().to_dict()}")
    
    # Step 2: Split data
    print("\n2️⃣ Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    
    # Step 3: Train models (if not already trained)
    print("\n3️⃣ Training models...")
    if not os.path.exists("models/final_model.joblib"):
        print("   No trained model found. Training new model...")
        train_main()
    else:
        print("   ✅ Trained model already exists at models/final_model.joblib")
    
    # Step 4: Evaluate model
    print("\n4️⃣ Evaluating model...")
    if os.path.exists("outputs/metrics.json"):
        print("   ✅ Evaluation results already exist")
    else:
        print("   Running evaluation...")
        evaluate_main()
    
    # Step 5: Generate explanations
    print("\n5️⃣ Generating model explanations...")
    if os.path.exists("outputs/feature_importance.png"):
        print("   ✅ Explainability analysis already exists")
    else:
        print("   Running explainability analysis...")
        explainability_main()
    
    # Step 6: Make sample predictions
    print("\n6️⃣ Making sample predictions...")
    import joblib
    model = joblib.load("models/final_model.joblib")
    
    # Use first few test samples
    sample_data = X_test.head(3)
    sample_labels = y_test.head(3)
    
    predictions = model.predict(sample_data)
    probabilities = model.predict_proba(sample_data)
    
    print("   Sample predictions:")
    for i in range(len(sample_data)):
        true_label = "Benign" if sample_labels.iloc[i] == 0 else "Malignant"
        pred_label = "Benign" if predictions[i] == 0 else "Malignant"
        confidence = max(probabilities[i])
        
        status = "✅" if predictions[i] == sample_labels.iloc[i] else "❌"
        print(f"   {status} Sample {i+1}: True={true_label}, Pred={pred_label}, Confidence={confidence:.1%}")
    
    # Step 7: Show available outputs
    print("\n7️⃣ Generated outputs:")
    outputs = [
        "models/final_model.joblib",
        "outputs/training_results.json",
        "outputs/metrics.json",
        "outputs/confusion_matrix.png",
        "outputs/roc_curve.png",
        "outputs/precision_recall_curve.png",
        "outputs/feature_importance.png",
        "outputs/shap/shap_summary.png"
    ]
    
    for output in outputs:
        if os.path.exists(output):
            print(f"   ✅ {output}")
        else:
            print(f"   ❌ {output}")
    
    print("\n🎉 Workflow demonstration completed!")
    print("\nNext steps:")
    print("1. 📊 Open notebooks/01_eda.ipynb for detailed data analysis")
    print("2. 🔬 Run 'streamlit run app/streamlit_app.py' for interactive predictions")
    print("3. 📈 Check outputs/ folder for visualizations and metrics")


def demonstrate_api_usage():
    """Demonstrate programmatic API usage."""
    
    print("\n" + "=" * 60)
    print("🔧 API Usage Examples")
    print("=" * 60)
    
    # Load model
    import joblib
    model = joblib.load("models/final_model.joblib")
    
    # Load sample data
    X, y = load_data(source="sklearn")
    
    print("\n📝 Making predictions programmatically:")
    
    # Example 1: Single prediction
    sample = X.iloc[[0]]  # First sample
    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0]
    
    print(f"\n1️⃣ Single prediction:")
    print(f"   Input shape: {sample.shape}")
    print(f"   Prediction: {prediction} ({'Benign' if prediction == 0 else 'Malignant'})")
    print(f"   Probabilities: Benign={probability[0]:.3f}, Malignant={probability[1]:.3f}")
    
    # Example 2: Batch predictions
    batch_samples = X.iloc[:5]  # First 5 samples
    batch_predictions = model.predict(batch_samples)
    batch_probabilities = model.predict_proba(batch_samples)
    
    print(f"\n2️⃣ Batch predictions:")
    print(f"   Input shape: {batch_samples.shape}")
    print(f"   Predictions: {batch_predictions}")
    print(f"   Max probabilities: {[max(prob) for prob in batch_probabilities]}")
    
    # Example 3: Custom input
    print(f"\n3️⃣ Custom input example:")
    print("   Creating sample with mean values...")
    
    custom_input = pd.DataFrame([X.mean()], columns=X.columns)
    custom_pred = model.predict(custom_input)[0]
    custom_prob = model.predict_proba(custom_input)[0]
    
    print(f"   Prediction: {custom_pred} ({'Benign' if custom_pred == 0 else 'Malignant'})")
    print(f"   Confidence: {max(custom_prob):.1%}")


if __name__ == "__main__":
    # Run the complete demonstration
    demonstrate_workflow()
    
    # Show API usage examples
    demonstrate_api_usage()
    
    print("\n" + "=" * 60)
    print("🚀 Ready to use!")
    print("=" * 60)
    print("\nTo run the Streamlit app:")
    print("   streamlit run app/streamlit_app.py")
    print("\nTo run individual components:")
    print("   python src/train.py        # Train models")
    print("   python src/evaluate.py     # Evaluate model")
    print("   python src/explainability.py  # Generate explanations")