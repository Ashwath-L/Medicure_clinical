import os
import sys
import joblib
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import HEART_FEATURE_COLS, STROKE_FEATURE_COLS
from src.explain import explain_heart_prediction, explain_stroke_prediction

# Global pipeline singletons to prevent reloading models per API call
_HEART_PIPELINE = None
_STROKE_PIPELINE = None

def get_models_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

def load_heart_pipeline():
    global _HEART_PIPELINE
    if _HEART_PIPELINE is None:
        path = os.path.join(get_models_dir(), 'heart_pipeline.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Heart pipeline artifact not found at: {path}")
        _HEART_PIPELINE = joblib.load(path)
    return _HEART_PIPELINE

def load_stroke_pipeline():
    global _STROKE_PIPELINE
    if _STROKE_PIPELINE is None:
        path = os.path.join(get_models_dir(), 'stroke_pipeline.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Stroke pipeline artifact not found at: {path}")
        _STROKE_PIPELINE = joblib.load(path)
    return _STROKE_PIPELINE

def predict_heart_sample(input_data: dict) -> dict:
    """
    Inference function for Heart Disease prediction.
    Accepts input dictionary, runs saved pipeline, extracts explanations,
    and returns structured JSON with safety disclaimers.
    """
    pipeline = load_heart_pipeline()
    
    # Format DataFrame with exact column order
    input_df = pd.DataFrame([input_data])[HEART_FEATURE_COLS]
    
    # Predict probability and class
    probs = pipeline.predict_proba(input_df)[0]
    pred_class = int(pipeline.predict(input_df)[0])
    prob = float(round(probs[1], 4))
    
    # Compute explainability
    important_features = explain_heart_prediction(pipeline, input_df, top_n=4)
    
    message = (
        "The model detected a higher-risk pattern in the provided clinical data."
        if pred_class == 1
        else "The model detected a lower-risk pattern in the provided clinical data."
    )
    
    return {
        "module": "heart",
        "prediction": pred_class,
        "probability": prob,
        "important_features": important_features,
        "message": message,
        "disclaimer": "This is an AI-assisted screening prototype and does not provide a medical diagnosis."
    }

def predict_stroke_sample(input_data: dict) -> dict:
    """
    Inference function for Stroke Risk prediction.
    Accepts input dictionary, runs saved pipeline, extracts explanations,
    and returns structured JSON with safety disclaimers.
    """
    pipeline = load_stroke_pipeline()
    
    # Format DataFrame with exact column order
    input_df = pd.DataFrame([input_data])[STROKE_FEATURE_COLS]
    
    # Predict probability and class
    probs = pipeline.predict_proba(input_df)[0]
    pred_class = int(pipeline.predict(input_df)[0])
    prob = float(round(probs[1], 4))
    
    # Compute explainability
    important_features = explain_stroke_prediction(pipeline, input_df, top_n=4)
    
    message = (
        "The model detected an elevated risk score pattern for stroke screening."
        if pred_class == 1
        else "The model detected a baseline risk score pattern for stroke screening."
    )
    
    return {
        "module": "stroke",
        "prediction": pred_class,
        "probability": prob,
        "important_features": important_features,
        "message": message,
        "disclaimer": "This is an AI-assisted screening prototype and does not provide a medical diagnosis."
    }
