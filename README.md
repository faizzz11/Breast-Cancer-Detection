# 🔬 Breast Cancer Detection Project

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url-here.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive machine learning project for breast cancer detection using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. This project implements multiple classification algorithms with proper preprocessing, feature selection, and model evaluation techniques.

## 🌐 **Live Demo**
**Try the app:** [Breast Cancer Detection App](https://your-app-url-here.streamlit.app)

## 🎯 **Key Results**
- **Accuracy:** 96.5%
- **ROC-AUC:** 99.5%
- **Precision:** 98.6%
- **Recall:** 95.8%

## 📊 Dataset Overview

The Wisconsin Diagnostic Breast Cancer dataset contains features computed from digitized images of fine needle aspirates (FNA) of breast masses. The dataset includes:

- **569 samples** (357 benign, 212 malignant)
- **30 features** describing cell nuclei characteristics
- **Binary classification** task (Benign vs Malignant)

### Features
Each sample has 30 features computed from 10 base measurements:
- **Radius** (mean distance from center to perimeter points)
- **Texture** (standard deviation of gray-scale values)
- **Perimeter**
- **Area**
- **Smoothness** (local variation in radius lengths)
- **Compactness** (perimeter² / area - 1.0)
- **Concavity** (severity of concave portions)
- **Concave points** (number of concave portions)
- **Symmetry**
- **Fractal dimension** ("coastline approximation" - 1)

For each base measurement, three statistics are computed:
- **Mean** (features 0-9)
- **Standard Error** (features 10-19)
- **Worst/Largest** (mean of three largest values, features 20-29)

## 🏗️ Project Structure

```
breast-cancer-detection/
├── data/
│   ├── raw/                    # Original dataset files
│   │   ├── wdbc.data          # Raw data file
│   │   └── wdbc.names         # Dataset description
│   └── processed/             # Processed datasets (generated)
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb # Data preprocessing
│   └── 03_modeling.ipynb     # Model training and evaluation
├── src/
│   ├── data_loader.py        # Data loading utilities
│   ├── preprocessing.py      # Preprocessing pipelines
│   └── models.py            # Model configurations
├── requirements.txt          # Python dependencies
└── README.md               # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd breast-cancer-detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Lab:**
   ```bash
   jupyter lab
   ```

### Quick Start

1. **Load the data:**
   ```python
   from src.data_loader import load_data
   
   # Load from sklearn (recommended)
   X, y = load_data(source="sklearn")
   
   # Or load from CSV file
   X, y = load_data(source="csv", path="data/raw/wdbc.data")
   ```

2. **Create preprocessing pipeline:**
   ```python
   from src.preprocessing import get_preprocessing_pipeline
   
   # Basic pipeline with scaling
   pipeline = get_preprocessing_pipeline(scaler="standard")
   
   # Pipeline with SMOTE for class imbalance
   pipeline = get_preprocessing_pipeline(scaler="standard", use_smote=True)
   ```

3. **Get pre-configured models:**
   ```python
   from src.models import get_classifiers
   
   models = get_classifiers(random_state=42)
   # Returns: LogisticRegression, RandomForest, SVM, XGBoost
   ```

## 📓 Notebooks Workflow

### 1. Exploratory Data Analysis (`01_eda.ipynb`)
- Dataset overview and quality assessment
- Target variable distribution analysis
- Feature distribution and correlation analysis
- Statistical significance testing
- Dimensionality reduction visualization
- Feature engineering insights

### 2. Data Preprocessing (`02_preprocessing.ipynb`)
- Feature scaling comparison (Standard, MinMax, Robust)
- Feature selection techniques (Univariate, RFE, Group-based)
- Class imbalance handling (SMOTE, undersampling, combined)
- Preprocessing pipeline creation and validation
- Data quality assessment after preprocessing

### 3. Machine Learning Modeling (`03_modeling.ipynb`)
- Baseline model evaluation
- Cross-validation analysis
- Hyperparameter tuning (Grid Search, Random Search)
- Comprehensive performance evaluation
- Model interpretability with SHAP
- Final model selection and validation

## 🤖 Available Models

The project includes four pre-configured classifiers:

| Model | Type | Strengths | Best For |
|-------|------|-----------|----------|
| **Logistic Regression** | Linear | Fast, interpretable, probabilistic | Baseline, linear relationships |
| **Random Forest** | Ensemble | Robust, feature importance, handles non-linearity | General purpose, feature analysis |
| **SVM** | Kernel-based | Effective in high dimensions, memory efficient | Small datasets, complex boundaries |
| **XGBoost** | Gradient Boosting | High performance, handles missing values | Maximum accuracy, competitions |

## 📈 Key Features

### Data Loading
- **Flexible data sources**: Load from sklearn or CSV files
- **Consistent interface**: Same API regardless of data source
- **Data validation**: Automatic consistency checks
- **Feature naming**: Proper feature names for interpretability

### Preprocessing
- **Multiple scaling options**: Standard, MinMax, Robust scalers
- **Feature selection**: Univariate, RFE, and group-based selection
- **Class imbalance handling**: SMOTE, undersampling, combined approaches
- **Pipeline integration**: sklearn-compatible preprocessing pipelines
- **Custom transformers**: Feature group selection and analysis

### Model Evaluation
- **Comprehensive metrics**: Accuracy, precision, recall, F1, AUC
- **Cross-validation**: Stratified k-fold for robust evaluation
- **Hyperparameter tuning**: Grid search and randomized search
- **Model interpretability**: SHAP values and feature importance
- **Calibration analysis**: Probability calibration assessment

## 🔧 Configuration Options

### Preprocessing Pipeline Options
```python
# Basic scaling only
pipeline = get_preprocessing_pipeline(scaler="standard")

# With SMOTE for class imbalance
pipeline = get_preprocessing_pipeline(scaler="robust", use_smote=True)

# With feature selection
pipeline = get_preprocessing_pipeline_with_selection(
    features="mean",  # or list of specific features
    scaler="standard",
    use_smote=True
)
```

### Model Hyperparameter Grids
```python
from src.models import get_hyperparameter_grids, get_quick_hyperparameter_grids

# Full hyperparameter grids
full_grids = get_hyperparameter_grids()

# Quick grids for faster experimentation
quick_grids = get_quick_hyperparameter_grids()
```

## 📊 Expected Results

Based on the literature and dataset characteristics, you can expect:

- **Baseline accuracy**: 85-90% with simple models
- **Optimized accuracy**: 95-98% with proper preprocessing and tuning
- **Key features**: Worst area, worst smoothness, mean texture typically most important
- **Class imbalance**: ~63% benign, ~37% malignant (moderate imbalance)

## 🏥 Clinical Relevance

This project demonstrates machine learning applications in medical diagnosis:

- **High sensitivity**: Minimize false negatives (missed cancer cases)
- **Balanced specificity**: Reduce false positives (unnecessary procedures)
- **Interpretability**: Understand which features drive predictions
- **Reliability**: Robust models with confidence estimates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Wisconsin Diagnostic Breast Cancer (WDBC) dataset from UCI ML Repository
- **Original creators**: Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian
- **Institution**: University of Wisconsin, Clinical Sciences Center

## 📚 References

1. W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology, volume 1905, pages 861-870, San Jose, CA, 1993.

2. O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and prognosis via linear programming. Operations Research, 43(4), pages 570-577, July-August 1995.

3. W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques to diagnose breast cancer from fine-needle aspirates. Cancer Letters 77 (1994) 163-171.

## 🚀 **Deployment**

### Deploy to Streamlit Community Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" and connect your GitHub repo
4. Set main file path to: `app/streamlit_app.py`
5. Click "Deploy!"

### Deploy to Hugging Face Spaces

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Create new Space with Streamlit SDK
3. Upload files or connect GitHub repo
4. Your app will be live instantly!

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/breast-cancer-detection.git
cd breast-cancer-detection

# Install dependencies
pip install -r requirements.txt

# Train the model (if not already trained)
python src/train.py

# Run the Streamlit app
streamlit run app/streamlit_app.py
```

## 📊 **Project Structure**

```
breast-cancer-detection/
├── app/
│   └── streamlit_app.py     # 🌐 Web application
├── src/
│   ├── train.py            # 🏋️ Model training
│   ├── evaluate.py         # 📊 Model evaluation
│   ├── explainability.py   # 🔍 Model interpretability
│   └── ...                 # Other utilities
├── models/
│   └── final_model.joblib  # 🤖 Trained model
├── outputs/                # 📈 Results and visualizations
├── notebooks/              # 📓 Jupyter notebooks
└── data/                   # 📁 Dataset
```

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ⚠️ **Disclaimer**

**Note**: This project is for educational and research purposes. Any clinical applications should involve medical professionals and proper validation studies.