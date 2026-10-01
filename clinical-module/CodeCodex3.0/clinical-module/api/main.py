import os
import sys
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator

# Include parent directory to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import (
    load_heart_pipeline,
    load_stroke_pipeline,
    predict_heart_sample,
    predict_stroke_sample
)

app = FastAPI(
    title="MedScope AI - Clinical Data Module API",
    description="Multimodal Medical AI Screening Prototype for Heart Disease and Stroke Risk Assessment.",
    version="1.0.0"
)

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Validation Error Handler to prevent stack trace leaks
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        loc = " -> ".join([str(l) for l in err.get("loc", [])])
        msg = err.get("msg", "Invalid value")
        errors.append(f"{loc}: {msg}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "details": errors,
            "message": "Provided input failed schema validation. Please check field types and allowed ranges."
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred during prediction. Stack trace masked for security."
        }
    )

# --- Pydantic Models ---

class HeartInput(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age in years (1-120)")
    sex: int = Field(..., ge=0, le=1, description="Sex (1 = Male, 0 = Female)")
    cp: int = Field(..., ge=0, le=3, description="Chest Pain Type (0-3)")
    trestbps: int = Field(..., ge=50, le=300, description="Resting Blood Pressure in mm Hg (50-300)")
    chol: int = Field(..., ge=80, le=700, description="Serum Cholesterol in mg/dl (80-700)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG Result (0-2)")
    thalach: int = Field(..., ge=40, le=250, description="Maximum Heart Rate Achieved (40-250)")
    exang: int = Field(..., ge=0, le=1, description="Exercise Induced Angina (1 = Yes, 0 = No)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST Depression (0.0-10.0)")
    slope: int = Field(..., ge=0, le=2, description="Slope of Peak Exercise ST Segment (0-2)")
    ca: int = Field(..., ge=0, le=4, description="Major Vessels Count (0-4)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia Result (0-3)")

class StrokeInput(BaseModel):
    gender: str = Field(..., description="Gender ('Male', 'Female', 'Other')")
    age: float = Field(..., ge=0.0, le=120.0, description="Age in years (0.0-120.0)")
    hypertension: int = Field(..., ge=0, le=1, description="Hypertension (1 = Yes, 0 = No)")
    heart_disease: int = Field(..., ge=0, le=1, description="Heart Disease History (1 = Yes, 0 = No)")
    ever_married: str = Field(..., description="Ever Married ('Yes', 'No')")
    work_type: str = Field(..., description="Work Type ('Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked')")
    Residence_type: str = Field(..., description="Residence Type ('Urban', 'Rural')")
    avg_glucose_level: float = Field(..., ge=40.0, le=350.0, description="Average Glucose Level (40.0-350.0)")
    bmi: Optional[float] = Field(None, ge=10.0, le=100.0, description="Body Mass Index (BMI, optional)")
    smoking_status: str = Field(..., description="Smoking Status ('formerly smoked', 'never smoked', 'smokes', 'Unknown')")

    @field_validator('gender')
    def validate_gender(cls, v):
        allowed = ['Male', 'Female', 'Other']
        if v not in allowed:
            raise ValueError(f"gender must be one of {allowed}")
        return v

    @field_validator('ever_married')
    def validate_ever_married(cls, v):
        allowed = ['Yes', 'No']
        if v not in allowed:
            raise ValueError(f"ever_married must be one of {allowed}")
        return v

    @field_validator('work_type')
    def validate_work_type(cls, v):
        allowed = ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
        if v not in allowed:
            raise ValueError(f"work_type must be one of {allowed}")
        return v

    @field_validator('Residence_type')
    def validate_residence_type(cls, v):
        allowed = ['Urban', 'Rural']
        if v not in allowed:
            raise ValueError(f"Residence_type must be one of {allowed}")
        return v

    @field_validator('smoking_status')
    def validate_smoking_status(cls, v):
        allowed = ['formerly smoked', 'never smoked', 'smokes', 'Unknown']
        if v not in allowed:
            raise ValueError(f"smoking_status must be one of {allowed}")
        return v

# --- Startup Event ---
@app.on_event("startup")
def startup_event():
    print("Pre-loading ML pipelines at startup...")
    try:
        load_heart_pipeline()
        load_stroke_pipeline()
        print("Both Heart and Stroke pipelines loaded successfully!")
    except Exception as e:
        print(f"Warning during model loading: {e}")

# --- API Endpoints ---

@app.get("/health")
def health_check():
    heart_status = "loaded"
    stroke_status = "loaded"
    try:
        load_heart_pipeline()
    except Exception:
        heart_status = "not_loaded"
        
    try:
        load_stroke_pipeline()
    except Exception:
        stroke_status = "not_loaded"
        
    return {
        "status": "ok",
        "module": "clinical",
        "models": {
            "heart": heart_status,
            "stroke": stroke_status
        }
    }

@app.post("/predict/heart")
def predict_heart(payload: HeartInput):
    try:
        result = predict_heart_sample(payload.model_dump())
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing heart prediction: {str(e)}"
        )

@app.post("/predict/stroke")
def predict_stroke(payload: StrokeInput):
    try:
        result = predict_stroke_sample(payload.model_dump())
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing stroke prediction: {str(e)}"
        )
