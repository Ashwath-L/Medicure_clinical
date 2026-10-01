import numpy as np
import pandas as pd

# Mapping of raw feature names to clean human-readable names for Heart
HEART_FEATURE_LABELS = {
    'age': 'Age',
    'sex': 'Sex',
    'cp': 'Chest Pain Type',
    'trestbps': 'Resting Blood Pressure',
    'chol': 'Serum Cholesterol',
    'fbs': 'Fasting Blood Sugar',
    'restecg': 'Resting ECG Result',
    'thalach': 'Maximum Heart Rate',
    'exang': 'Exercise Induced Angina',
    'oldpeak': 'ST Depression (oldpeak)',
    'slope': 'ST Segment Slope',
    'ca': 'Major Vessels Count',
    'thal': 'Thalassemia Result'
}

# Mapping of raw feature names to clean human-readable names for Stroke
STROKE_FEATURE_LABELS = {
    'age': 'Age',
    'avg_glucose_level': 'Average Glucose Level',
    'bmi': 'Body Mass Index (BMI)',
    'hypertension': 'Hypertension',
    'heart_disease': 'Heart Disease History',
    'gender': 'Gender',
    'ever_married': 'Marital Status',
    'work_type': 'Work Type',
    'Residence_type': 'Residence Type',
    'smoking_status': 'Smoking Status'
}

STROKE_CATEGORICAL_COLS = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

def explain_heart_prediction(pipeline, input_df: pd.DataFrame, top_n: int = 4) -> list:
    """
    Computes feature contributions for a Heart Logistic Regression prediction.
    Calculates contribution = scaled_value * model_coefficient.
    Returns the top N model-influencing features with safety-compliant wording.
    """
    scaler = pipeline.named_steps['scaler']
    model = pipeline.named_steps['classifier']
    
    feature_names = list(input_df.columns)
    scaled_values = scaler.transform(input_df)[0]
    coefs = model.coef_[0]
    raw_values = input_df.iloc[0].to_dict()
    
    contributions = []
    for idx, col in enumerate(feature_names):
        c = scaled_values[idx] * coefs[idx]
        label = HEART_FEATURE_LABELS.get(col, col)
        raw_val = raw_values[col]
        contributions.append({
            'feature_key': col,
            'feature_name': label,
            'raw_value': raw_val,
            'contribution': float(c),
            'abs_contribution': float(abs(c))
        })
        
    contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)
    top_features = contributions[:top_n]
    
    formatted_explanations = []
    for item in top_features:
        name = item['feature_name']
        val = item['raw_value']
        if item['contribution'] > 0:
            effect = "increases model risk score"
        else:
            effect = "lowers model risk score"
            
        formatted_explanations.append({
            "feature": name,
            "value": str(val),
            "effect": effect,
            "description": f"{name} ({val}) {effect}"
        })
        
    return formatted_explanations

def explain_stroke_prediction(pipeline, input_df: pd.DataFrame, top_n: int = 4) -> list:
    """
    Computes aggregated feature contributions for a Stroke Logistic Regression prediction.
    Handles ColumnTransformer (StandardScaler for numerical, OneHotEncoder for categorical).
    Maps transformed features back to original feature names.
    Returns top N model-influencing features with safety-compliant wording.
    """
    preprocessor = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['classifier']
    
    transformed_X = preprocessor.transform(input_df)[0]
    feature_names_out = list(preprocessor.get_feature_names_out())
    coefs = model.coef_[0]
    
    aggregated_contributions = {}
    
    for idx, fname in enumerate(feature_names_out):
        c = transformed_X[idx] * coefs[idx]
        
        if fname.startswith('num__'):
            orig_col = fname.replace('num__', '')
        elif fname.startswith('cat__'):
            cat_str = fname.replace('cat__', '')
            orig_col = cat_str
            for col in STROKE_CATEGORICAL_COLS:
                if cat_str.startswith(col + '_'):
                    orig_col = col
                    break
        else:
            orig_col = fname
            
        if orig_col not in aggregated_contributions:
            aggregated_contributions[orig_col] = 0.0
        aggregated_contributions[orig_col] += c
        
    raw_values = input_df.iloc[0].to_dict()
    
    explanations_list = []
    for orig_col, total_c in aggregated_contributions.items():
        label = STROKE_FEATURE_LABELS.get(orig_col, orig_col)
        raw_val = raw_values.get(orig_col, 'N/A')
        explanations_list.append({
            'feature_key': orig_col,
            'feature_name': label,
            'raw_value': raw_val,
            'contribution': float(total_c),
            'abs_contribution': float(abs(total_c))
        })
        
    explanations_list.sort(key=lambda x: x['abs_contribution'], reverse=True)
    top_features = explanations_list[:top_n]
    
    formatted_explanations = []
    for item in top_features:
        name = item['feature_name']
        val = item['raw_value']
        if item['contribution'] > 0:
            effect = "increases model risk score"
        else:
            effect = "lowers model risk score"
            
        formatted_explanations.append({
            "feature": name,
            "value": str(val),
            "effect": effect,
            "description": f"{name} ({val}) {effect}"
        })
        
    return formatted_explanations
