import pandas as pd
import numpy as np
import hashlib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Feature column names for the Heart dataset
HEART_FEATURE_COLS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]
HEART_TARGET_COL = 'target'

# Feature column names for the Stroke dataset
STROKE_NUMERICAL_COLS = ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease']
STROKE_CATEGORICAL_COLS = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
STROKE_FEATURE_COLS = STROKE_NUMERICAL_COLS + STROKE_CATEGORICAL_COLS
STROKE_TARGET_COL = 'stroke'

def create_heart_groups(df: pd.DataFrame, feature_cols: list = None) -> pd.Series:
    """
    Creates a deterministic group identifier for each row based ONLY on feature columns.
    Identical feature rows will receive the exact same group_id.
    Target is strictly excluded from group generation.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != HEART_TARGET_COL]
        
    def hash_row(row):
        val_str = "_".join(str(row[col]) for col in feature_cols)
        return hashlib.md5(val_str.encode('utf-8')).hexdigest()
        
    group_ids = df[feature_cols].apply(hash_row, axis=1)
    unique_hashes = group_ids.unique()
    hash_to_id = {h: i for i, h in enumerate(unique_hashes)}
    
    return group_ids.map(hash_to_id)

def build_heart_preprocessing_pipeline() -> Pipeline:
    """
    Builds a preprocessing pipeline for the Heart dataset.
    """
    preprocessor = Pipeline([
        ('scaler', StandardScaler())
    ])
    return preprocessor

def build_stroke_preprocessor() -> ColumnTransformer:
    """
    Builds a ColumnTransformer preprocessor for the Stroke dataset:
    - Numerical features (age, avg_glucose_level, bmi, hypertension, heart_disease): SimpleImputer(median) + StandardScaler
    - Categorical features (gender, ever_married, work_type, Residence_type, smoking_status): OneHotEncoder(handle_unknown='ignore')
    Preventing data leakage by placing imputer, scaling, and one-hot encoding strictly inside pipeline.
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, STROKE_NUMERICAL_COLS),
            ('cat', cat_pipeline, STROKE_CATEGORICAL_COLS)
        ],
        remainder='drop'
    )
    
    return preprocessor
