
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# 1. Define project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student_data.csv"
MODEL_PATH = BASE_DIR / "model.pkl"


# 2. Check dataset availability
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Check that student_data.csv exists inside backend/data."
    )


# 3. Load dataset
df = pd.read_csv(DATA_PATH)

print("\n========== DATASET ==========")
print(df.head())
print("\nDataset shape:", df.shape)


# 4. Validate dataset
required_columns = ["study_hours", "marks"]

if not all(column in df.columns for column in required_columns):
    raise ValueError(
        "CSV must contain columns: study_hours, marks"
    )

df = df[required_columns].copy()

df["study_hours"] = pd.to_numeric(
    df["study_hours"], errors="coerce"
)
df["marks"] = pd.to_numeric(
    df["marks"], errors="coerce"
)

df = df.dropna()

if len(df) < 5:
    raise ValueError("At least 5 valid rows are required.")

if (df["study_hours"] < 0).any():
    raise ValueError("Study hours cannot be negative.")

if not df["study_hours"].between(0, 24).all():
    raise ValueError("Study hours must be between 0 and 24.")

if not df["marks"].between(0, 100).all():
    raise ValueError("Marks must be between 0 and 100.")

print("\nValid dataset rows:", len(df))


# 5. Separate input and target
X = df[["study_hours"]]
y = df["marks"]


# 6. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 7. Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)


# 8. Evaluate model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n========== MODEL EVALUATION ==========")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("MAE:", round(mae, 2))
print("R2 Score:", round(r2, 2))


# 9. Compare actual and predicted marks
comparison = pd.DataFrame({
    "Study Hours": X_test["study_hours"].to_numpy(),
    "Actual Marks": y_test.to_numpy(),
    "Predicted Marks": predictions.round(2)
})

comparison["Absolute Error"] = (
    comparison["Actual Marks"] - comparison["Predicted Marks"]
).abs().round(2)

print("\n========== ACTUAL VS PREDICTED ==========")
print(comparison.to_string(index=False))


# 10. Show example predictions
sample_hours = pd.DataFrame({
    "study_hours": [2, 4, 6, 8, 10]
})

sample_predictions = model.predict(sample_hours)

sample_results = pd.DataFrame({
    "Study Hours": sample_hours["study_hours"],
    "Predicted Marks": sample_predictions.round(2)
})

print("\n========== SAMPLE PREDICTIONS ==========")
print(sample_results.to_string(index=False))


# 11. Save trained model
joblib.dump(model, MODEL_PATH)

print("\n========== MODEL SAVED ==========")
print("Model trained successfully!")
print("Model saved at:", MODEL_PATH)
