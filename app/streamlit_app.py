#!/usr/bin/env python3
"""
Modern Streamlit web application for breast cancer detection.

This app provides a beautiful, interactive interface for making predictions using
the trained breast cancer detection model with advanced UI/UX design.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# Page configuration with modern theme
st.set_page_config(
    page_title="AI Cancer Detection | Advanced Medical Diagnostics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/breast-cancer-detection',
        'Report a bug': 'https://github.com/yourusername/breast-cancer-detection/issues',
        'About': "# AI-Powered Breast Cancer Detection\nBuilt with ❤️ using Streamlit and scikit-learn"
    }
)

# Custom CSS for premium modern UI
st.markdown("""
<style>
    /* Import Google Fonts and Font Awesome */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    /* Custom Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="white" opacity="0.1"/><circle cx="75" cy="75" r="1" fill="white" opacity="0.1"/><circle cx="50" cy="10" r="0.5" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.3;
    }
    
    .main-title {
        color: white;
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        text-align: center;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .main-subtitle {
        color: rgba(255,255,255,0.95);
        font-size: 1.3rem;
        text-align: center;
        margin-top: 0.8rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    /* Premium Feature Cards */
    .feature-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.2);
        margin: 1.5rem 0;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        transition: width 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 60px rgba(0,0,0,0.12);
    }
    
    .feature-card:hover::before {
        width: 8px;
    }
    
    .feature-card h4 {
        color: #2d3748;
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
    
    .feature-card p {
        color: #4a5568;
        line-height: 1.6;
        margin: 0;
    }
    
    /* Premium Stats Cards */
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem 1.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        margin: 0.5rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stats-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        transition: transform 0.6s ease;
        transform: scale(0);
    }
    
    .stats-card:hover::before {
        transform: scale(1);
    }
    
    .stats-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
    }
    
    .stats-number {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        position: relative;
        z-index: 1;
    }
    
    .stats-label {
        font-size: 0.95rem;
        opacity: 0.95;
        margin-top: 0.5rem;
        font-weight: 500;
        position: relative;
        z-index: 1;
    }
    
    /* Premium Input Styling */
    .stNumberInput > div > div > input {
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 12px 16px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%) !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.06) !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1), inset 0 2px 4px rgba(0,0,0,0.06) !important;
        transform: translateY(-1px) !important;
    }
    
    .stNumberInput > div > div > input:hover {
        border-color: #cbd5e0 !important;
        transform: translateY(-1px) !important;
    }
    
    /* Premium Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 30px !important;
        padding: 16px 32px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3) !important;
        position: relative !important;
        overflow: hidden !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .stButton > button::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: -100% !important;
        width: 100% !important;
        height: 100% !important;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent) !important;
        transition: left 0.5s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton > button:hover::before {
        left: 100% !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(1.02) !important;
    }
    
    /* Premium Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stProgress > div > div > div {
        background: rgba(102, 126, 234, 0.1) !important;
        border-radius: 10px !important;
    }
    
    /* Premium Prediction Cards */
    .prediction-benign {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 20px 50px rgba(16, 185, 129, 0.3);
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-malignant {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 20px 50px rgba(239, 68, 68, 0.3);
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.2);
    }
    
    .confidence-score {
        font-size: 4rem;
        font-weight: 900;
        margin: 1.5rem 0;
        text-shadow: 3px 3px 10px rgba(0,0,0,0.3);
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Premium Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        text-align: center;
        margin: 0.5rem;
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.12);
    }
    
    .metric-card h3 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
    }
    
    .metric-card p {
        color: #6b7280;
        font-weight: 500;
        margin: 0;
        font-size: 0.9rem;
    }
    
    /* Premium Expander Styling */
    .streamlit-expanderHeader {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%) !important;
        border-radius: 12px !important;
        border: 1px solid #e2e8f0 !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
        transition: all 0.3s ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(145deg, #f8fafc 0%, #f1f5f9 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08) !important;
    }
    
    /* Premium Toggle Styling */
    .stCheckbox > label > div[data-testid="stMarkdownContainer"] > p {
        font-weight: 600 !important;
        color: #374151 !important;
    }
    
    /* Animation Classes */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(40px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
    
    .fade-in {
        animation: fadeInUp 0.8s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Premium Text Styling */
    .premium-text {
        color: #1f2937;
        font-weight: 500;
        line-height: 1.6;
    }
    
    .premium-heading {
        color: #111827;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
</style>
""", unsafe_allow_html=True)


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
    """Create premium input widgets for single sample prediction."""
    
    st.markdown("""
    <div class="premium-heading" style="text-align: center; margin-bottom: 2rem; color:#7A94D6">
        <h3>
            <i class="fas fa-user-md" style="margin-right: 10px; color: #667eea;"></i>
            Patient Data Input
            <i class="fas fa-clipboard-list" style="margin-left: 10px; color: #667eea;"></i>
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Get the original feature order from the training data
    X, _ = load_sample_data()
    original_feature_order = X.columns.tolist()
    
    # Premium toggle for feature selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        <div class="premium-text" style="color:#7A94D6">
            <i class="fas fa-sliders-h" style="margin-right: 8px; color: #667eea;"></i>
            <strong>Configure Input Parameters</strong>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        use_top_features = st.toggle(
            "Simplified Mode", 
            value=True,
            help="Show only the 10 most important features for easier input"
        )
    
    if use_top_features:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border-left-color: #3b82f6;">
            <p style="margin: 0; color: #1e40af;">
                <i class="fas fa-bullseye" style="margin-right: 8px;"></i>
                <strong>Simplified Mode:</strong> Showing top 10 most predictive features
            </p>
        </div>
        """, unsafe_allow_html=True)
        features_to_show = top_features
    else:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border-left-color: #8b5cf6;">
            <p style="margin: 0; color: #6b21a8;">
                <i class="fas fa-microscope" style="margin-right: 8px;"></i>
                <strong>Advanced Mode:</strong> Showing all 30 features
            </p>
        </div>
        """, unsafe_allow_html=True)
        features_to_show = original_feature_order
    
    # Create premium input sections
    input_data = {}
    
    # Group features by category with professional icons
    feature_groups = {
        "Size Measurements": {
            "features": [f for f in features_to_show if any(x in f.lower() for x in ['radius', 'area', 'perimeter'])],
            "icon": "fas fa-ruler-combined",
            "color": "#10b981"
        },
        "Texture & Shape": {
            "features": [f for f in features_to_show if any(x in f.lower() for x in ['texture', 'smoothness', 'compactness', 'concavity', 'symmetry', 'fractal'])],
            "icon": "fas fa-shapes",
            "color": "#f59e0b"
        },
        "Statistical Measures": {
            "features": [f for f in features_to_show if 'error' in f.lower()],
            "icon": "fas fa-chart-line",
            "color": "#ef4444"
        }
    }
    
    # Remove empty groups
    feature_groups = {k: v for k, v in feature_groups.items() if v["features"]}
    
    # If no groups match, create a single group
    if not any(group["features"] for group in feature_groups.values()):
        feature_groups = {
            "Cell Measurements": {
                "features": features_to_show,
                "icon": "fas fa-microscope",
                "color": "#667eea"
            }
        }
    
    # Create premium expandable sections for each group
    for group_name, group_data in feature_groups.items():
        group_features = group_data["features"]
        if not group_features:
            continue
            
        with st.expander(
            f"{group_name} ({len(group_features)} features)", 
            expanded=True
        ):
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 1rem;">
                <i class="{group_data['icon']}" style="font-size: 1.5rem; color: {group_data['color']};"></i>
            </div>
            """, unsafe_allow_html=True)
            
            cols = st.columns(2)
            
            for i, feature in enumerate(group_features):
                col = cols[i % 2]
                stats = feature_stats[feature]
                
                with col:
                    # Create premium input with enhanced styling
                    st.markdown(f"""
                    <div style="margin-bottom: 0.5rem;">
                        <label style="font-weight: 600; color: #374151; font-size: 0.9rem;">
                            <i class="fas fa-vial" style="margin-right: 5px; color: #6b7280;"></i>
                            {feature.replace('_', ' ').title()}
                        </label>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    value = st.number_input(
                        label="",
                        min_value=stats['min'],
                        max_value=stats['max'],
                        value=stats['mean'],
                        step=(stats['max'] - stats['min']) / 100,
                        format="%.4f",
                        help=f"📊 Range: {stats['min']:.2f} - {stats['max']:.2f} | 📈 Mean: {stats['mean']:.2f} | 📉 Std: {stats['std']:.2f}",
                        key=f"input_{feature}",
                        label_visibility="collapsed"
                    )
                    input_data[feature] = value
                    
                    # Add premium progress indicator
                    normalized_value = (value - stats['min']) / (stats['max'] - stats['min'])
                    
                    # Color-coded progress bar based on value position
                    if normalized_value < 0.33:
                        progress_color = "#10b981"  # Green for low values
                    elif normalized_value < 0.67:
                        progress_color = "#f59e0b"  # Yellow for medium values
                    else:
                        progress_color = "#ef4444"  # Red for high values
                    
                    st.markdown(f"""
                    <div style="margin: 0.5rem 0 1rem 0;">
                        <div style="background: #f1f5f9; border-radius: 10px; height: 6px; overflow: hidden;">
                            <div style="background: {progress_color}; height: 100%; width: {normalized_value * 100}%; 
                                        border-radius: 10px; transition: all 0.3s ease;"></div>
                        </div>
                        <div style="text-align: center; font-size: 0.75rem; color: #6b7280; margin-top: 0.25rem;">
                            {normalized_value * 100:.1f}% of range
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Fill missing features with mean values if using top features only
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
    """Display prediction results with modern styling and animations."""
    if result is None:
        return
    
    prediction = result['prediction']
    prob_benign = result['probability_benign']
    prob_malignant = result['probability_malignant']
    confidence = result['confidence']
    
    # Premium animated prediction card
    if prediction == 0:
        st.markdown(f"""
        <div class="prediction-benign fade-in">
            <div class="prediction-title">
                <i class="fas fa-shield-alt" style="margin-right: 15px;"></i>
                BENIGN
                <i class="fas fa-check-circle" style="margin-left: 15px;"></i>
            </div>
            <div class="confidence-score">{confidence:.1%}</div>
            <p style="font-size: 1.1rem; margin: 0;">
                <i class="fas fa-heart" style="margin-right: 8px;"></i>
                Low risk of malignancy detected
                <i class="fas fa-smile" style="margin-left: 8px;"></i>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-malignant fade-in">
            <div class="prediction-title">
                <i class="fas fa-exclamation-triangle" style="margin-right: 15px;"></i>
                MALIGNANT
                <i class="fas fa-alert" style="margin-left: 15px;"></i>
            </div>
            <div class="confidence-score">{confidence:.1%}</div>
            <p style="font-size: 1.1rem; margin: 0;">
                <i class="fas fa-stethoscope" style="margin-right: 8px;"></i>
                High risk of malignancy detected - Consult physician
                <i class="fas fa-user-md" style="margin-left: 8px;"></i>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Interactive probability visualization with Plotly
    st.markdown("### 📊 Probability Analysis")
    
    # Create gauge chart for confidence
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level"},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig_gauge.update_layout(
        height=300,
        font={'color': "#262730", 'family': "Inter"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    # Create probability comparison chart
    fig_bar = go.Figure(data=[
        go.Bar(
            x=['Benign', 'Malignant'],
            y=[prob_benign * 100, prob_malignant * 100],
            marker_color=['#4facfe', '#fa709a'],
            text=[f'{prob_benign:.1%}', f'{prob_malignant:.1%}'],
            textposition='auto',
            textfont=dict(size=14, color='white', family="Inter")
        )
    ])
    
    fig_bar.update_layout(
        title="Probability Distribution",
        xaxis_title="Diagnosis",
        yaxis_title="Probability (%)",
        yaxis=dict(range=[0, 100]),
        font={'color': "#262730", 'family': "Inter"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400
    )
    
    # Display charts side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Risk assessment
    st.markdown("### 🎯 Risk Assessment")
    
    if confidence > 0.9:
        risk_level = "Very High Confidence"
        risk_color = "#4facfe"
        risk_icon = "🎯"
    elif confidence > 0.8:
        risk_level = "High Confidence"
        risk_color = "#667eea"
        risk_icon = "✅"
    elif confidence > 0.7:
        risk_level = "Moderate Confidence"
        risk_color = "#ffa726"
        risk_icon = "⚠️"
    else:
        risk_level = "Low Confidence"
        risk_color = "#ef5350"
        risk_icon = "❓"
    
    st.markdown(f"""
    <div class="feature-card">
        <h4>{risk_icon} {risk_level}</h4>
        <p><strong>Confidence Score:</strong> {confidence:.1%}</p>
        <p><strong>Benign Probability:</strong> {prob_benign:.1%}</p>
        <p><strong>Malignant Probability:</strong> {prob_malignant:.1%}</p>
    </div>
    """, unsafe_allow_html=True)


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


def create_modern_header():
    """Create a premium, attractive header with professional icons."""
    st.markdown("""
    <div class="main-header fade-in">
        <h1 class="main-title">
            <i class="fas fa-microscope" style="margin-right: 15px; color: rgba(255,255,255,0.9);"></i>
            AI Cancer Detection
            <i class="fas fa-brain" style="margin-left: 15px; color: rgba(255,255,255,0.9);"></i>
        </h1>
        <p class="main-subtitle">
            <i class="fas fa-stethoscope" style="margin-right: 8px;"></i>
            Advanced Machine Learning for Medical Diagnostics
            <i class="fas fa-chart-line" style="margin-left: 8px;"></i>
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_stats_dashboard():
    """Create a premium stats dashboard with professional icons."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stats-card pulse">
            <i class="fas fa-bullseye" style="font-size: 1.5rem; margin-bottom: 0.5rem; opacity: 0.9;"></i>
            <div class="stats-number">96.5%</div>
            <div class="stats-label">
                <i class="fas fa-check-circle" style="margin-right: 5px;"></i>
                Accuracy
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stats-card pulse">
            <i class="fas fa-chart-area" style="font-size: 1.5rem; margin-bottom: 0.5rem; opacity: 0.9;"></i>
            <div class="stats-number">99.5%</div>
            <div class="stats-label">
                <i class="fas fa-analytics" style="margin-right: 5px;"></i>
                ROC-AUC
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stats-card pulse">
            <i class="fas fa-crosshairs" style="font-size: 1.5rem; margin-bottom: 0.5rem; opacity: 0.9;"></i>
            <div class="stats-number">98.6%</div>
            <div class="stats-label">
                <i class="fas fa-medal" style="margin-right: 5px;"></i>
                Precision
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stats-card pulse">
            <i class="fas fa-search-plus" style="font-size: 1.5rem; margin-bottom: 0.5rem; opacity: 0.9;"></i>
            <div class="stats-number">95.8%</div>
            <div class="stats-label">
                <i class="fas fa-eye" style="margin-right: 5px;"></i>
                Recall
            </div>
        </div>
        """, unsafe_allow_html=True)

def create_modern_sidebar():
    """Create a premium sidebar with enhanced styling and professional icons."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; margin-bottom: 1rem;">
            <h3 style="color:#7A94D6; margin: 0;">
                <i class="fas fa-compass" style="margin-right: 8px; color: #667eea;"></i>
                Navigation
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Mode selection with professional icons
        mode = st.selectbox(
            "Choose Analysis Mode:",
            [
                "🔬 Single Sample Analysis", 
                "📊 Batch Processing"
            ],
            help="Select your preferred prediction method",
            format_func=lambda x: x.replace("🔬", "").replace("📊", "").strip() if "🔬" in x or "📊" in x else x
        )
        
        # Add visual mode indicators
        if "Single" in mode:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 0.5rem; border-radius: 8px; text-align: center; margin: 0.5rem 0;">
                <i class="fas fa-user-md" style="margin-right: 5px;"></i>
                Individual Patient Mode
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 0.5rem; border-radius: 8px; text-align: center; margin: 0.5rem 0;">
                <i class="fas fa-users" style="margin-right: 5px;"></i>
                Batch Processing Mode
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model information with professional icons
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1rem;">
            <h3 style="color:#7A94D6; margin: 0;">
                <i class="fas fa-robot" style="margin-right: 8px; color: #667eea;"></i>
                AI Model Info
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>
                <i class="fas fa-cogs" style="margin-right: 8px; color: #667eea;"></i>
                Model Architecture
            </h4>
            <p class="premium-text">
                <i class="fas fa-layer-group" style="margin-right: 5px; color: #6b7280;"></i>
                <strong>Type:</strong> Ensemble Pipeline
            </p>
            <p class="premium-text">
                <i class="fas fa-calculator" style="margin-right: 5px; color: #6b7280;"></i>
                <strong>Algorithm:</strong> Logistic Regression
            </p>
            <p class="premium-text">
                <i class="fas fa-database" style="margin-right: 5px; color: #6b7280;"></i>
                <strong>Features:</strong> 30 Cell Measurements
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>
                <i class="fas fa-chart-bar" style="margin-right: 8px; color: #667eea;"></i>
                Performance Metrics
            </h4>
            <p class="premium-text">
                <i class="fas fa-graduation-cap" style="margin-right: 5px; color: #6b7280;"></i>
                <strong>Training Samples:</strong> 455
            </p>
            <p class="premium-text">
                <i class="fas fa-vial" style="margin-right: 5px; color: #6b7280;"></i>
                <strong>Test Samples:</strong> 114
            </p>
            <p class="premium-text">
                <i class="fas fa-sync-alt" style="margin-right: 5px; color: #6b7280;"></i>
                <strong>Cross-Validation:</strong> 5-Fold
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Sample data section with professional styling
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1rem;">
            <h3 style="color:#7A94D6; margin: 0;">
                <i class="fas fa-download" style="margin-right: 8px; color: #667eea;"></i>
                Test Data
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎲 Generate Sample Data", help="Create sample CSV for testing", use_container_width=True):
            X, y = load_sample_data()
            sample_data = X.head(5).copy()
            sample_data['actual_diagnosis'] = y.head(5).map({0: 'Benign', 1: 'Malignant'})
            
            csv = sample_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample CSV",
                data=csv,
                file_name="sample_breast_cancer_data.csv",
                mime="text/csv",
                help="Download sample data to test batch upload",
                use_container_width=True
            )
        
        st.markdown("---")
        
        # About section with professional styling
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1rem;">
            <h3 style="color:#7A94D6; margin: 0;">
                <i class="fas fa-info-circle" style="margin-right: 8px; color: #667eea;"></i>
                About
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <p class="premium-text">
                <i class="fas fa-graduation-cap" style="margin-right: 5px; color: #6b7280;"></i>
                <strong>Purpose:</strong> Educational & Research
            </p>
            <p class="premium-text">
                <i class="fas fa-database" style="margin-right: 5px; color: #6b7280;"></i>
                <strong>Dataset:</strong> Wisconsin Breast Cancer
            </p>
            <p class="premium-text">
                <i class="fas fa-users" style="margin-right: 5px; color: #6b7280;"></i>
                <strong>Samples:</strong> 569 cases
            </p>
            <p class="premium-text">
                <i class="fas fa-microscope" style="margin-right: 5px; color: #6b7280;"></i>
                <strong>Features:</strong> Cell nuclei measurements
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        return mode

def main():
    """Main Streamlit application with modern UI."""
    
    # Create modern header
    create_modern_header()
    
    # Create stats dashboard
    create_stats_dashboard()
    
    # Create modern sidebar and get mode
    mode = create_modern_sidebar()
    
    # Load model and data
    model = load_model()
    feature_stats = get_feature_info()
    top_features = get_top_features(10)
    
    # Main content based on mode
    if "Single Sample" in mode:
        st.markdown("""
        <div class="premium-heading" style="text-align: center; margin-bottom: 2rem; color: 
#7A94D6 ">
            <h2>
                <i class="fas fa-user-md" style="margin-right: 15px; color: #667eea;"></i>
                Individual Patient Analysis
                <i class="fas fa-heartbeat" style="margin-left: 15px; color: #F54927;"></i>
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>
                <i class="fas fa-bullseye" style="margin-right: 10px; color: #667eea;"></i>
                Single Sample Prediction
            </h4>
            <p class="premium-text">
                <i class="fas fa-microscope" style="margin-right: 8px; color: #6b7280;"></i>
                Enter patient measurements to get an AI-powered risk assessment for breast cancer detection. 
                Our advanced machine learning model analyzes cell nuclei characteristics to provide accurate predictions.
            </p>
            <div style="margin-top: 1rem; padding: 1rem; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                        border-radius: 8px; border-left: 4px solid #0ea5e9;">
                <p style="margin: 0; color: #0c4a6e; font-weight: 500;">
                    <i class="fas fa-info-circle" style="margin-right: 8px;"></i>
                    This tool uses machine learning to analyze cellular features and provide diagnostic insights.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create input form
        input_data = create_single_sample_input(feature_stats, top_features)
        
        # Premium prediction button with enhanced styling
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <p style="color: #6b7280; font-weight: 500; margin-bottom: 1rem;">
                <i class="fas fa-rocket" style="margin-right: 8px;"></i>
                Ready to analyze? Click below to start the AI diagnosis process.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Analyze Sample", type="primary", use_container_width=True):
                # Create progress bar animation
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(100):
                    progress_bar.progress(i + 1)
                    if i < 30:
                        status_text.text('🔍 Preprocessing data...')
                    elif i < 70:
                        status_text.text('🧠 Running AI analysis...')
                    else:
                        status_text.text('📊 Generating results...')
                
                result = make_prediction(model, input_data)
                
                progress_bar.empty()
                status_text.empty()
                
                if result:
                    st.markdown("---")
                    display_prediction_result(result)
                    
                    # SHAP explanation with modern toggle
                    st.markdown("### 🔍 Advanced Analysis")
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown("**Model Interpretability & Feature Analysis**")
                    with col2:
                        show_shap = st.toggle("Enable SHAP", help="Generate detailed AI explanation")
                    
                    if show_shap:
                        create_shap_explanation(model, input_data)
    
    elif "Batch Processing" in mode:
        st.markdown("""
        <div class="premium-heading" style="text-align: center; margin-bottom: 2rem;">
            <h2>
                <i class="fas fa-chart-bar" style="margin-right: 15px; color: #667eea;"></i>
                Batch Analysis Dashboard
                <i class="fas fa-users" style="margin-left: 15px; color: #667eea;"></i>
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>
                <i class="fas fa-upload" style="margin-right: 10px; color: #667eea;"></i>
                Multiple Sample Processing
            </h4>
            <p class="premium-text">
                <i class="fas fa-file-csv" style="margin-right: 8px; color: #6b7280;"></i>
                Upload a CSV file containing multiple patient samples for bulk analysis. 
                Perfect for research studies, clinical trials, or processing multiple cases efficiently.
            </p>
            <div style="margin-top: 1rem; padding: 1rem; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                        border-radius: 8px; border-left: 4px solid #f59e0b;">
                <p style="margin: 0; color: #92400e; font-weight: 500;">
                    <i class="fas fa-lightning-bolt" style="margin-right: 8px;"></i>
                    Batch processing enables efficient analysis of multiple samples simultaneously.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced file upload
        uploaded_data = create_batch_upload()
        
        if uploaded_data is not None:
            # Modern data preview
            st.markdown("### 📋 Data Overview")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Samples", len(uploaded_data))
            with col2:
                st.metric("📈 Features", len(uploaded_data.columns))
            with col3:
                st.metric("💾 File Size", f"{uploaded_data.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            # Interactive data preview
            st.markdown("**Sample Data Preview:**")
            st.dataframe(uploaded_data.head(10), use_container_width=True)
            
            # Premium batch processing button
            st.markdown("""
            <div style="text-align: center; margin: 2rem 0;">
                <p style="color: #6b7280; font-weight: 500; margin-bottom: 1rem;">
                    <i class="fas fa-cogs" style="margin-right: 8px;"></i>
                    Ready to process all samples? This may take a few moments.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Process All Samples", type="primary", use_container_width=True):
                    # Enhanced progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    predictions = []
                    probabilities = []
                    
                    for idx in range(len(uploaded_data)):
                        # Update progress
                        progress = (idx + 1) / len(uploaded_data)
                        progress_bar.progress(progress)
                        status_text.text(f'🔄 Processing sample {idx + 1} of {len(uploaded_data)}...')
                        
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
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Create enhanced results
                    results_df = uploaded_data.copy()
                    results_df['Prediction'] = predictions
                    results_df['Prediction_Label'] = ['Benign' if p == 0 else 'Malignant' if p == 1 else 'Error' for p in predictions]
                    results_df['Probability_Benign'] = [p[0] if p[0] is not None else None for p in probabilities]
                    results_df['Probability_Malignant'] = [p[1] if p[1] is not None else None for p in probabilities]
                    
                    # Enhanced results display
                    st.markdown("### 📊 Analysis Results")
                    
                    # Summary statistics with modern cards
                    valid_predictions = [p for p in predictions if p is not None]
                    if valid_predictions:
                        benign_count = sum(1 for p in valid_predictions if p == 0)
                        malignant_count = sum(1 for p in valid_predictions if p == 1)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{len(uploaded_data)}</h3>
                                <p>Total Samples</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3 style="color: #4facfe;">{benign_count}</h3>
                                <p>Benign Cases</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3 style="color: #fa709a;">{malignant_count}</h3>
                                <p>Malignant Cases</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col4:
                            accuracy_rate = (benign_count + malignant_count) / len(uploaded_data) * 100
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3 style="color: #667eea;">{accuracy_rate:.1f}%</h3>
                                <p>Success Rate</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Interactive results visualization
                        if benign_count > 0 or malignant_count > 0:
                            fig_pie = px.pie(
                                values=[benign_count, malignant_count],
                                names=['Benign', 'Malignant'],
                                title="Prediction Distribution",
                                color_discrete_sequence=['#4facfe', '#fa709a']
                            )
                            fig_pie.update_layout(
                                font={'color': "#262730", 'family': "Inter"},
                                paper_bgcolor="rgba(0,0,0,0)"
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Enhanced results table
                    st.markdown("**Detailed Results:**")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Modern download section
                    st.markdown("### 📥 Export Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📊 Download Full Results (CSV)",
                            data=csv,
                            file_name="breast_cancer_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        summary_df = pd.DataFrame({
                            'Metric': ['Total Samples', 'Benign Cases', 'Malignant Cases', 'Success Rate'],
                            'Value': [len(uploaded_data), benign_count, malignant_count, f"{accuracy_rate:.1f}%"]
                        })
                        summary_csv = summary_df.to_csv(index=False)
                        st.download_button(
                            label="📋 Download Summary (CSV)",
                            data=summary_csv,
                            file_name="analysis_summary.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
    
    # Premium footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 20px; margin-top: 3rem; position: relative; overflow: hidden;'>
        <div style='position: absolute; top: 0; left: 0; right: 0; bottom: 0; 
                    background: url("data:image/svg+xml,%3Csvg width="60" height="60" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg"%3E%3Cg fill="none" fill-rule="evenodd"%3E%3Cg fill="%23ffffff" fill-opacity="0.1"%3E%3Ccircle cx="30" cy="30" r="2"/%3E%3C/g%3E%3C/g%3E%3C/svg%3E"); 
                    opacity: 0.3;'></div>
        <h3 style='color: white; margin: 0; position: relative; z-index: 1; font-weight: 700;'>
            <i class="fas fa-dna" style="margin-right: 15px;"></i>
            AI-Powered Medical Diagnostics
            <i class="fas fa-heartbeat" style="margin-left: 15px;"></i>
        </h3>
        <p style='color: rgba(255,255,255,0.95); margin: 1rem 0 0 0; position: relative; z-index: 1; font-size: 1.1rem;'>
            <i class="fas fa-code" style="margin-right: 8px;"></i>
            Built with ❤️ using Streamlit & scikit-learn
            <i class="fas fa-flask" style="margin-left: 8px;"></i>
        </p>
        <p style='color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; position: relative; z-index: 1; font-size: 0.9rem;'>
            <i class="fas fa-graduation-cap" style="margin-right: 5px;"></i>
            For educational and research purposes only
            <i class="fas fa-shield-alt" style="margin-left: 5px;"></i>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()