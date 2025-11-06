"""
Preprocessing utilities for the breast cancer detection project.

This module provides functions to create sklearn-compatible preprocessing pipelines
with scaling and optional SMOTE for handling class imbalance.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin


def train_test_split_data(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform stratified train-test split on the data.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        test_size (float): Proportion of dataset to include in test split
        random_state (int): Random state for reproducibility
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
            X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )


def build_preprocessing_pipeline(
    numeric_features: List[str],
    scaler: str = "standard",
    use_robust: bool = False,
    use_smote: bool = False,
    smote_random_state: int = 42
) -> ImbPipeline:
    """
    Build a comprehensive preprocessing pipeline suitable for GridSearchCV.
    
    This pipeline includes imputation, scaling, and optional SMOTE oversampling.
    The pipeline is designed to work with imblearn's Pipeline which properly
    handles SMOTE during cross-validation by only applying it to training folds.
    
    Args:
        numeric_features (List[str]): List of numeric feature names
        scaler (str): Type of scaler - "standard" or "minmax"
        use_robust (bool): If True, use RobustScaler instead of StandardScaler
        use_smote (bool): Whether to apply SMOTE for oversampling minority class
        smote_random_state (int): Random state for SMOTE reproducibility
    
    Returns:
        ImbPipeline: Complete preprocessing pipeline
    
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.model_selection import GridSearchCV
        >>> 
        >>> # Build preprocessing pipeline
        >>> preprocessor = build_preprocessing_pipeline(
        ...     numeric_features=X.columns.tolist(),
        ...     scaler="standard",
        ...     use_smote=True
        ... )
        >>> 
        >>> # Create full pipeline with classifier
        >>> full_pipeline = ImbPipeline([
        ...     ('preprocessor', preprocessor),
        ...     ('classifier', RandomForestClassifier(random_state=42))
        ... ])
        >>> 
        >>> # Use in GridSearchCV
        >>> param_grid = {
        ...     'classifier__n_estimators': [50, 100],
        ...     'classifier__max_depth': [None, 10]
        ... }
        >>> grid_search = GridSearchCV(full_pipeline, param_grid, cv=5)
        >>> grid_search.fit(X_train, y_train)
    """
    steps = []
    
    # Step 1: Imputation for missing values
    steps.append(('imputer', SimpleImputer(strategy='median')))
    
    # Step 2: Scaling
    if use_robust:
        scaler_obj = RobustScaler()
    elif scaler == "standard":
        scaler_obj = StandardScaler()
    elif scaler == "minmax":
        scaler_obj = MinMaxScaler()
    else:
        raise ValueError(f"Scaler '{scaler}' not supported. Choose 'standard' or 'minmax'")
    
    steps.append(('scaler', scaler_obj))
    
    # Step 3: Optional SMOTE (only applied during training)
    if use_smote:
        steps.append(('smote', SMOTE(random_state=smote_random_state)))
    
    return ImbPipeline(steps)


def get_preprocessing_pipeline(
    scaler: str = "standard", 
    use_smote: bool = False,
    smote_random_state: int = 42
) -> Union[Pipeline, ImbPipeline]:
    """
    Create a preprocessing pipeline with scaling and optional SMOTE.
    
    Args:
        scaler (str): Type of scaler to use. Options:
            - "standard": StandardScaler (mean=0, std=1)
            - "minmax": MinMaxScaler (range 0-1)
            - "robust": RobustScaler (median and IQR)
        use_smote (bool): Whether to apply SMOTE for oversampling minority class
        smote_random_state (int): Random state for SMOTE reproducibility
    
    Returns:
        Union[Pipeline, ImbPipeline]: Preprocessing pipeline
            - Pipeline if use_smote=False
            - ImbPipeline if use_smote=True (from imbalanced-learn)
    
    Raises:
        ValueError: If scaler type is not supported
    """
    # Get the scaler
    scaler_obj = _get_scaler(scaler)
    
    if use_smote:
        # Use imbalanced-learn Pipeline for SMOTE compatibility
        steps = [
            ('scaler', scaler_obj),
            ('smote', SMOTE(random_state=smote_random_state))
        ]
        return ImbPipeline(steps)
    else:
        # Use regular sklearn Pipeline
        steps = [
            ('scaler', scaler_obj)
        ]
        return Pipeline(steps)


def _get_scaler(scaler_type: str) -> BaseEstimator:
    """
    Get the appropriate scaler object.
    
    Args:
        scaler_type (str): Type of scaler
    
    Returns:
        BaseEstimator: Scaler object
    
    Raises:
        ValueError: If scaler type is not supported
    """
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }
    
    if scaler_type not in scalers:
        raise ValueError(f"Scaler '{scaler_type}' not supported. "
                        f"Choose from: {list(scalers.keys())}")
    
    return scalers[scaler_type]


def create_feature_groups() -> dict:
    """
    Create feature groups for the Wisconsin Breast Cancer dataset.
    
    Groups features by their type (mean, standard error, worst) and
    by their measurement category.
    
    Returns:
        dict: Dictionary with feature group mappings
    """
    # Use sklearn feature names format
    base_features = [
        'radius', 'texture', 'perimeter', 'area', 'smoothness',
        'compactness', 'concavity', 'concave points', 'symmetry', 'fractal dimension'
    ]
    
    # Group by statistic type (using sklearn naming convention)
    mean_features = [f'mean {feature}' for feature in base_features]
    se_features = [f'{feature} error' for feature in base_features]
    worst_features = [f'worst {feature}' for feature in base_features]
    
    # Group by measurement category
    size_features = []
    for stat_prefix in ['mean', 'worst']:
        for feature in ['radius', 'perimeter', 'area']:
            size_features.append(f'{stat_prefix} {feature}')
    for feature in ['radius', 'perimeter', 'area']:
        size_features.append(f'{feature} error')
    
    texture_features = []
    for stat_prefix in ['mean', 'worst']:
        for feature in ['texture', 'smoothness', 'symmetry', 'fractal dimension']:
            texture_features.append(f'{stat_prefix} {feature}')
    for feature in ['texture', 'smoothness', 'symmetry', 'fractal dimension']:
        texture_features.append(f'{feature} error')
    
    shape_features = []
    for stat_prefix in ['mean', 'worst']:
        for feature in ['compactness', 'concavity', 'concave points']:
            shape_features.append(f'{stat_prefix} {feature}')
    for feature in ['compactness', 'concavity', 'concave points']:
        shape_features.append(f'{feature} error')
    
    return {
        'by_statistic': {
            'mean': mean_features,
            'se': se_features,
            'worst': worst_features
        },
        'by_category': {
            'size': size_features,
            'texture': texture_features,
            'shape': shape_features
        },
        'all_features': mean_features + se_features + worst_features
    }


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom transformer for selecting specific features or feature groups.
    
    This can be useful for feature selection experiments or focusing on
    specific types of measurements.
    """
    
    def __init__(self, features: Union[list, str] = 'all'):
        """
        Initialize feature selector.
        
        Args:
            features (Union[list, str]): Features to select
                - list: Specific feature names
                - str: Feature group name ('mean', 'se', 'worst', 'size', 'texture', 'shape', 'all')
        """
        self.features = features
        self.selected_features_ = None
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the feature selector.
        
        Args:
            X (pd.DataFrame): Input features
            y: Target (ignored)
        
        Returns:
            self: Fitted transformer
        """
        if isinstance(self.features, list):
            self.selected_features_ = self.features
        elif isinstance(self.features, str):
            feature_groups = create_feature_groups()
            
            if self.features == 'all':
                self.selected_features_ = list(X.columns)
            elif self.features in feature_groups['by_statistic']:
                self.selected_features_ = feature_groups['by_statistic'][self.features]
            elif self.features in feature_groups['by_category']:
                self.selected_features_ = feature_groups['by_category'][self.features]
            else:
                raise ValueError(f"Unknown feature group: {self.features}")
        else:
            raise ValueError("Features must be a list or string")
        
        # Validate that all selected features exist in X
        missing_features = set(self.selected_features_) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features not found in data: {missing_features}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by selecting features.
        
        Args:
            X (pd.DataFrame): Input features
        
        Returns:
            pd.DataFrame: Transformed data with selected features
        """
        if self.selected_features_ is None:
            raise ValueError("Transformer not fitted yet")
        
        return X[self.selected_features_]


def get_preprocessing_pipeline_with_selection(
    features: Union[list, str] = 'all',
    scaler: str = "standard", 
    use_smote: bool = False,
    smote_random_state: int = 42
) -> Union[Pipeline, ImbPipeline]:
    """
    Create a preprocessing pipeline with feature selection, scaling, and optional SMOTE.
    
    Args:
        features (Union[list, str]): Features to select (see FeatureSelector)
        scaler (str): Type of scaler to use
        use_smote (bool): Whether to apply SMOTE
        smote_random_state (int): Random state for SMOTE
    
    Returns:
        Union[Pipeline, ImbPipeline]: Complete preprocessing pipeline
    """
    # Get the scaler
    scaler_obj = _get_scaler(scaler)
    
    if use_smote:
        # Use imbalanced-learn Pipeline for SMOTE compatibility
        steps = [
            ('feature_selector', FeatureSelector(features)),
            ('scaler', scaler_obj),
            ('smote', SMOTE(random_state=smote_random_state))
        ]
        return ImbPipeline(steps)
    else:
        # Use regular sklearn Pipeline
        steps = [
            ('feature_selector', FeatureSelector(features)),
            ('scaler', scaler_obj)
        ]
        return Pipeline(steps)


def analyze_class_imbalance(y: pd.Series) -> dict:
    """
    Analyze class imbalance in the target variable.
    
    Args:
        y (pd.Series): Target variable
    
    Returns:
        dict: Class imbalance analysis
    """
    class_counts = y.value_counts().sort_index()
    total_samples = len(y)
    
    analysis = {
        'class_counts': class_counts.to_dict(),
        'class_percentages': (class_counts / total_samples * 100).to_dict(),
        'imbalance_ratio': class_counts.max() / class_counts.min(),
        'minority_class': class_counts.idxmin(),
        'majority_class': class_counts.idxmax(),
        'is_imbalanced': (class_counts.max() / class_counts.min()) > 1.5
    }
    
    return analysis


if __name__ == "__main__":
    # Example usage
    from data_loader import load_data
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    
    print("Loading data...")
    X, y = load_data(source="sklearn")
    
    print("\nTesting train-test split...")
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shape: X={X_test.shape}, y={y_test.shape}")
    
    print("\nTesting preprocessing pipeline...")
    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(
        numeric_features=X.columns.tolist(),
        scaler="standard",
        use_smote=True
    )
    
    # Test preprocessing pipeline
    print("Fitting preprocessing pipeline...")
    X_processed, y_processed = preprocessor.fit_resample(X_train, y_train)
    print(f"Processed data shape: X={X_processed.shape}, y={y_processed.shape}")
    
    # Create and test classifier separately
    classifier = RandomForestClassifier(random_state=42, n_estimators=50)
    classifier.fit(X_processed, y_processed)
    
    # Transform test data (without SMOTE)
    X_test_processed = preprocessor.named_steps['scaler'].transform(
        preprocessor.named_steps['imputer'].transform(X_test)
    )
    score = classifier.score(X_test_processed, y_test)
    print(f"Pipeline accuracy: {score:.3f}")
    
    print("\nAnalyzing class imbalance...")
    imbalance_info = analyze_class_imbalance(y)
    for key, value in imbalance_info.items():
        print(f"{key}: {value}")
    
    print("\nTesting legacy pipelines...")
    
    # Test basic pipeline
    pipeline_basic = get_preprocessing_pipeline(scaler="standard", use_smote=False)
    X_transformed = pipeline_basic.fit_transform(X)
    print(f"Basic pipeline output shape: {X_transformed.shape}")
    
    # Test pipeline with SMOTE
    pipeline_smote = get_preprocessing_pipeline(scaler="standard", use_smote=True)
    X_smote, y_smote = pipeline_smote.fit_resample(X, y)
    print(f"SMOTE pipeline output shape: X={X_smote.shape}, y={y_smote.shape}")
    
    print("\nFeature groups:")
    feature_groups = create_feature_groups()
    for group_type, groups in feature_groups.items():
        print(f"{group_type}:")
        if isinstance(groups, dict):
            for subgroup, features in groups.items():
                print(f"  {subgroup}: {len(features)} features")
        else:
            print(f"  {len(groups)} features")