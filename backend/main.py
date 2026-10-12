
from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

# Load trained ML model
model = joblib.load(MODEL_PATH)

# Create FastAPI app
app = FastAPI(
    title="Student Marks Prediction API",
    description="Predict student marks using study hours.",
    version="1.0.0"
)

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request data format
class StudentData(BaseModel):
    study_hours: float = Field(ge=0, le=24)


# Home route
@app.get("/")
def home():
    return {
        "message": "Student Marks Prediction API is running!"
    }


# Prediction route
@app.post("/predict")
def predict_marks(data: StudentData):
    prediction = model.predict([[data.study_hours]])[0]

    return {
        "study_hours": data.study_hours,
        "predicted_marks": round(float(prediction), 2)
    }
