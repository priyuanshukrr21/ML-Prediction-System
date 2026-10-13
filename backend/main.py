
from pathlib import Path
import os
import sqlite3
import hmac

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
DATABASE_PATH = BASE_DIR / "visitor_data.db"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. Run backend/train_model.py first."
    )

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Student Marks Prediction API",
    description="Student marks prediction and visitor records API.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your frontend domain when deployed.
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Admin-Token"],
)


# Create database and table automatically.
def init_database():
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS visitors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                location TEXT NOT NULL,
                visited_at TEXT NOT NULL DEFAULT
                    (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
            )
        """)


init_database()


# Request models
class StudentData(BaseModel):
    study_hours: float = Field(ge=0, le=24)


class VisitorData(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    location: str = Field(min_length=2, max_length=120)
    consent: bool


@app.get("/")
def home():
    return {"message": "Student Marks Prediction API is running!"}


# Existing prediction API — preserved.
@app.post("/predict")
def predict_marks(data: StudentData):
    input_data = pd.DataFrame({
        "study_hours": [data.study_hours]
    })

    prediction = float(model.predict(input_data)[0])
    prediction = max(0.0, min(100.0, prediction))

    return {
        "study_hours": data.study_hours,
        "predicted_marks": round(prediction, 2),
        "message": "Prediction generated successfully",
    }


# Save visitor details after consent.
@app.post("/visitors")
def save_visitor(data: VisitorData):
    if not data.consent:
        raise HTTPException(
            status_code=400,
            detail="Consent is required to save visitor details.",
        )

    name = " ".join(data.name.split()).strip()
    location = " ".join(data.location.split()).strip()

    if len(name) < 2 or len(location) < 2:
        raise HTTPException(
            status_code=422,
            detail="Please enter a valid name and location.",
        )

    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.execute(
            "INSERT INTO visitors (name, location) VALUES (?, ?)",
            (name, location),
        )
        visitor_id = cursor.lastrowid

    return {
        "message": "Visitor details saved successfully.",
        "visitor_id": visitor_id,
    }


# Admin-only endpoint to read visitor records.
# Set ADMIN_TOKEN in your environment before using this endpoint.
@app.get("/admin/visitors")
def get_visitors(x_admin_token: str = Header(default="")):
    admin_token = os.getenv("ADMIN_TOKEN", "")

    if not admin_token:
        raise HTTPException(
            status_code=503,
            detail="Admin access is not configured on the server.",
        )

    if not hmac.compare_digest(x_admin_token, admin_token):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized. Admin access required.",
        )

    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT id, name, location, visited_at
            FROM visitors
            ORDER BY id DESC
        """).fetchall()

    return {
        "total_visitors": len(rows),
        "visitors": [dict(row) for row in rows],
    }
