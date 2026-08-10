from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
import os
# def replace_zeros_with_nan_df(X, cols):
#     X = X.copy()
#     for col in cols:
#         if col in X.columns:
#             X[col] = X[col].replace(0, np.nan)
#     return X
# zero_as_missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
app = FastAPI()

# Base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load models
heart_model = joblib.load(os.path.join(BASE_DIR, "models", "heart_disease_model.pkl"))
diabetes_model = joblib.load(os.path.join(BASE_DIR, "models", "diabetes_model.pkl"))
from pydantic import BaseModel,Field

class HeartInput(BaseModel):
   age: int = Field(..., ge=1, le=120, description="Age of patient")
   sex: int = Field(..., ge=0, le=1, description="0 = Female, 1 = Male")
   cp: int = Field(..., ge=0, le=3, description="Chest pain type (0–3)")
   trestbps: int = Field(..., ge=80, le=200, description="Resting blood pressure")
   chol: int = Field(..., ge=100, le=600, description="Cholesterol level")
   fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar >120 mg/dl (1 = True, 0 = False)")
   restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0–2)")
   thalach: int = Field(..., ge=60, le=220, description="Maximum heart rate achieved")
   exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (1 = Yes, 0 = No)")
   oldpeak: float = Field(..., ge=0, le=10, description="ST depression")
   slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment")
   ca: int = Field(..., ge=0, le=4, description="Number of major vessels (0–4)")
   thal: int = Field(..., ge=0, le=3, description="Thalassemia (0–3)")


# Home route
@app.get("/")
def home():
    return {"message": "Multi Disease Prediction API is running 🚀"}


# ❤️ Heart Disease Prediction
@app.post("/predict/heart")
def predict_heart(data: HeartInput):
    try:
        values = list(data.model_dump().values())
        input_data = np.array([values])
        
        prediction = heart_model.predict(input_data)[0]
        
        return {
            "disease": "heart",
            "prediction": int(prediction)
        }
    except Exception as e:
        return {"error": str(e)}


# 🩺 Diabetes Prediction
class DiabetesInput(BaseModel):
    Pregnancies: int=Field(...,ge=0,le=20)
    Glucose: float = Field(..., ge=0, le=300)
    BloodPressure: float = Field(..., ge=0, le=200)
    SkinThickness: float = Field(..., ge=0, le=100)
    Insulin: float = Field(..., ge=0, le=900)
    BMI: float = Field(..., ge=0, le=70)
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=3)
    Age: int = Field(..., ge=1, le=120)
@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesInput):
    try:
        values = list(data.model_dump().values())
        columns = [
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age"
                ]
        input_data = pd.DataFrame([values],columns=columns)
        # input_scaled=scaler.transform(input_data)
        
        prediction = diabetes_model.predict(input_data)[0]
        
        return {
            "disease": "diabetes",
            "prediction": int(prediction)
        }
    except Exception as e:
        return {"error": str(e)}