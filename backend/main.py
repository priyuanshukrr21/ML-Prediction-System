
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"


# 2. Check and load trained model
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found at: {MODEL_PATH}. "
        "Please run backend/train_model.py first."
    )

model = joblib.load(MODEL_PATH)


# 3. Create FastAPI application
app = FastAPI(
    title="Student Marks Prediction API",
    description="Predict student marks using study hours.",
    version="1.0.0"
)


# 4. Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 5. Validate incoming data
class StudentData(BaseModel):
    study_hours: float = Field(
        ge=0,
        le=24,
        description="Daily study hours, between 0 and 24"
    )


# 6. Home route
@app.get("/")
def home():
    return {
        "message": "Student Marks Prediction API is running!"
    }


# 7. Prediction route
@app.post("/predict")
def predict_marks(data: StudentData):
    # Prepare input in the same format used during training
    input_data = pd.DataFrame({
        "study_hours": [data.study_hours]
    })

    # Predict marks
    prediction = float(model.predict(input_data)[0])

    # Keep predicted marks within the 0–100 range
    prediction = max(0.0, min(100.0, prediction))

    return {
        "study_hours": data.study_hours,
        "predicted_marks": round(prediction, 2),
        "message": "Prediction generated successfully"
    }
