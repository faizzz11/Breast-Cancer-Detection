#!/usr/bin/env python3
"""
Streamlit web application for breast cancer detection.

This app provides an interactive interface for making predictions using
the trained breast cancer detection model.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from data_loader import load_data


# Page configuration
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model():
    """Load the trained model (cached for performance)."""
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'final_model.joblib')
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}")
        st.error("Please run 'python src/train.py' first to train a model.")
        st.stop()
    
    return joblib.load(model_path)


@st.cache_data
def load_sample_data():
    """Load sample data for reference (cached for performance)."""
    X, y = load_data(source="sklearn")
    return X, y


def get_feature_info() -> Dict[str, Dict[str, float]]:
    """Get feature statistics for input validation."""
    X, _ = load_sample_data()
    
    feature_stats = {}
    for col in X.columns:
        feature_stats[col] = {
            'min': float(X[col].min()),
            'max': float(X[col].max()),
            'mean': float(X[col].mean()),
            'std': float(X[col].std())
        }
    
    return feature_stats


def get_top_features(n: int = 10) -> List[str]:
    """Get the top N most important features based on correlation with target."""
    X, y = load_sample_data()
    
    # Calculate correlation with target
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    
    return correlations.head(n).index.tolist()


def create_single_sample_input(feature_stats: Dict[str, Dict[str, float]], 
                              top_features: List[str]) -> pd.DataFrame:
    """Create input widgets for single sample prediction."""
    st.subheader("📝 Enter Feature Values")
    
    # Get the original feature order from the training data
    X, _ = load_sample_data()
    original_feature_order = X.columns.tolist()
    
    # Option to use top features only
    use_top_features = st.checkbox(
        f"Use only top {len(top_features)} most important features", 
        value=True,
        help="This simplifies input by showing only the most predictive features"
    )
    
    features_to_show = top_features if use_top_features else original_feature_order
    
    # Create input widgets
    input_data = {}
    
    # Organize features in columns for better layout
    cols = st.columns(2)
    
    for i, feature in enumerate(features_to_show):
        col = cols[i % 2]
        
        stats = feature_stats[feature]
        
        with col:
            # Create input widget with reasonable defaults
            value = st.number_input(
                label=feature,
                min_value=stats['min'],
                max_value=stats['max'],
                value=stats['mean'],
                step=(stats['max'] - stats['min']) / 100,
                format="%.4f",
                help=f"Range: {stats['min']:.2f} - {stats['max']:.2f}, Mean: {stats['mean']:.2f}"
            )
            input_data[feature] = value
    
    # Fill missing features with mean values if using top features only
    # IMPORTANT: Maintain the original feature order
    complete_input_data = {}
    for feature in original_feature_order:
        if feature in input_data:
            complete_input_data[feature] = input_data[feature]
        else:
            complete_input_data[feature] = feature_stats[feature]['mean']
    
    # Create DataFrame with features in the correct order
    input_df = pd.DataFrame([complete_input_data], columns=original_feature_order)
    
    return input_df


def create_batch_upload() -> Optional[pd.DataFrame]:
    """Create file upload widget for batch predictions."""
    st.subheader("📁 Upload CSV File")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with feature columns matching the training data"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Get the original feature order from training data
            X, _ = load_sample_data()
            expected_columns = X.columns.tolist()  # Maintain order
            expected_columns_set = set(expected_columns)
            uploaded_columns = set(df.columns)
            
            missing_columns = expected_columns_set - uploaded_columns
            extra_columns = uploaded_columns - expected_columns_set
            
            if missing_columns:
                st.warning(f"⚠️ Missing columns: {list(missing_columns)}")
                
                # Option to fill missing columns with mean values
                if st.button("Fill missing columns with mean values"):
                    feature_stats = get_feature_info()
                    for col in missing_columns:
                        df[col] = feature_stats[col]['mean']
                    st.success("✅ Missing columns filled with mean values")
                    st.rerun()
            
            if extra_columns:
                st.info(f"ℹ️ Extra columns will be ignored: {list(extra_columns)}")
            
            if not missing_columns:
                # Reorder columns to match training data order and keep only expected columns
                df_ordered = df[expected_columns]
                st.success(f"✅ File uploaded successfully! {len(df_ordered)} samples loaded.")
                return df_ordered
            
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    return None


def make_prediction(model, input_data: pd.DataFrame) -> Dict:
    """Make prediction using the trained model."""
    try:
        # Ensure columns are in the correct order
        X_train, _ = load_sample_data()
        expected_features = X_train.columns.tolist()
        
        # Reorder columns to match training data if needed
        if list(input_data.columns) != expected_features:
            input_data = input_data[expected_features]
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        return {
            'prediction': int(prediction),
            'probability_benign': float(probability[0]),
            'probability_malignant': float(probability[1]),
            'confidence': float(max(probability))
        }
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        
        # Debug information for troubleshooting
        with st.expander("🔍 Debug Information"):
            try:
                X_train, _ = load_sample_data()
                st.write(f"Expected features ({len(X_train.columns)}): {list(X_train.columns)}")
                st.write(f"Input features ({len(input_data.columns)}): {list(input_data.columns)}")
                st.write(f"Input data shape: {input_data.shape}")
                st.write(f"Model type: {type(model)}")
            except Exception as debug_e:
                st.write(f"Debug error: {debug_e}")
        
        return None


def display_prediction_result(result: Dict):
    """Display prediction results with styling."""
    if result is None:
        return
    
    prediction = result['prediction']
    prob_benign = result['probability_benign']
    prob_malignant = result['probability_malignant']
    confidence = result['confidence']
    
    # Main prediction display
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 0:
            st.success("🟢 **BENIGN**")
            st.write(f"**Confidence:** {confidence:.1%}")
        else:
            st.error("🔴 **MALIGNANT**")
            st.write(f"**Confidence:** {confidence:.1%}")
    
    with col2:
        # Probability breakdown
        st.write("**Probability Breakdown:**")
        st.write(f"• Benign: {prob_benign:.1%}")
        st.write(f"• Malignant: {prob_malignant:.1%}")
    
    # Probability bar chart
    fig, ax = plt.subplots(figsize=(8, 3))
    categories = ['Benign', 'Malignant']
    probabilities = [prob_benign, prob_malignant]
    colors = ['lightgreen', 'lightcoral']
    
    bars = ax.bar(categories, probabilities, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Probabilities')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, prob in zip(bars, probabilities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def create_shap_explanation(model, input_data: pd.DataFrame):
    """Create SHAP explanation for the prediction."""
    if not SHAP_AVAILABLE:
        st.info("ℹ️ SHAP explanations not available. Install shap package for detailed explanations.")
        return
    
    try:
        st.subheader("🔍 Prediction Explanation (SHAP)")
        
        with st.spinner("Generating SHAP explanation..."):
            # Get the classifier from the pipeline
            classifier = model.named_steps.get('classifier', model)
            
            # Transform the data through preprocessing steps
            X_transformed = input_data.copy()
            for step_name, transformer in model.named_steps.items():
                if step_name != 'classifier' and step_name != 'smote':
                    if hasattr(transformer, 'transform'):
                        X_transformed = transformer.transform(X_transformed)
            
            # Convert to numpy array if needed
            if isinstance(X_transformed, pd.DataFrame):
                X_transformed = X_transformed.values
            
            # Create explainer (simplified for web app)
            explainer = shap.Explainer(classifier, X_transformed)
            shap_values = explainer(X_transformed)
            
            # Create SHAP plot
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)
            plt.close()
            
    except Exception as e:
        st.warning(f"⚠️ Could not generate SHAP explanation: {str(e)}")


def main():
    """Main Streamlit application."""
    # Header
    st.title("🔬 Breast Cancer Detection")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    mode = st.sidebar.selectbox(
        "Choose prediction mode:",
        ["Single Sample", "Batch Upload"],
        help="Select how you want to make predictions"
    )
    
    # Load model and data
    model = load_model()
    feature_stats = get_feature_info()
    top_features = get_top_features(10)
    
    # Model info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ Model Information")
    st.sidebar.write("**Model Type:** Ensemble Pipeline")
    st.sidebar.write("**Features:** 30 cell nuclei measurements")
    st.sidebar.write("**Classes:** Benign (0) / Malignant (1)")
    st.sidebar.write("**Accuracy:** 96.5%")
    st.sidebar.write("**ROC-AUC:** 99.5%")
    
    # Sample data download
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Test Data")
    if st.sidebar.button("Generate Sample CSV"):
        X, y = load_sample_data()
        sample_data = X.head(5).copy()
        sample_data['actual_diagnosis'] = y.head(5).map({0: 'Benign', 1: 'Malignant'})
        
        csv = sample_data.to_csv(index=False)
        st.sidebar.download_button(
            label="📥 Download Sample Data",
            data=csv,
            file_name="sample_breast_cancer_data.csv",
            mime="text/csv",
            help="Download sample data to test batch upload"
        )
    
    # Main content based on mode
    if mode == "Single Sample":
        st.header("🔬 Single Sample Prediction")
        st.write("Enter the feature values below to get a prediction for a single sample.")
        
        # Create input form
        input_data = create_single_sample_input(feature_stats, top_features)
        
        # Prediction button
        if st.button("🔮 Make Prediction", type="primary"):
            with st.spinner("Making prediction..."):
                result = make_prediction(model, input_data)
                
                if result:
                    st.markdown("---")
                    st.subheader("📊 Prediction Result")
                    display_prediction_result(result)
                    
                    # SHAP explanation
                    if st.checkbox("Show detailed explanation", help="Generate SHAP explanation for this prediction"):
                        create_shap_explanation(model, input_data)
    
    elif mode == "Batch Upload":
        st.header("📁 Batch Prediction")
        st.write("Upload a CSV file with multiple samples to get predictions for all of them.")
        
        # File upload
        uploaded_data = create_batch_upload()
        
        if uploaded_data is not None:
            # Show data preview
            st.subheader("📋 Data Preview")
            st.dataframe(uploaded_data.head())
            
            # Make predictions button
            if st.button("🔮 Make Predictions", type="primary"):
                with st.spinner(f"Making predictions for {len(uploaded_data)} samples..."):
                    
                    # Make predictions for all samples
                    predictions = []
                    probabilities = []
                    
                    for idx in range(len(uploaded_data)):
                        sample = uploaded_data.iloc[[idx]]
                        result = make_prediction(model, sample)
                        
                        if result:
                            predictions.append(result['prediction'])
                            probabilities.append([
                                result['probability_benign'],
                                result['probability_malignant']
                            ])
                        else:
                            predictions.append(None)
                            probabilities.append([None, None])
                    
                    # Create results dataframe
                    results_df = uploaded_data.copy()
                    results_df['Prediction'] = predictions
                    results_df['Prediction_Label'] = ['Benign' if p == 0 else 'Malignant' if p == 1 else 'Error' for p in predictions]
                    results_df['Probability_Benign'] = [p[0] if p[0] is not None else None for p in probabilities]
                    results_df['Probability_Malignant'] = [p[1] if p[1] is not None else None for p in probabilities]
                    
                    # Display results
                    st.subheader("📊 Prediction Results")
                    
                    # Summary statistics
                    valid_predictions = [p for p in predictions if p is not None]
                    if valid_predictions:
                        benign_count = sum(1 for p in valid_predictions if p == 0)
                        malignant_count = sum(1 for p in valid_predictions if p == 1)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Samples", len(uploaded_data))
                        with col2:
                            st.metric("Benign", benign_count)
                        with col3:
                            st.metric("Malignant", malignant_count)
                    
                    # Results table
                    st.dataframe(results_df)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="breast_cancer_predictions.csv",
                        mime="text/csv"
                    )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>🔬 Breast Cancer Detection Model | Built with Streamlit</p>
        <p><em>For educational and research purposes only. Not for clinical use.</em></p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()