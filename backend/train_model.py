from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student_data.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

# Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print(df.head())

# Prepare data
X = df[["study_hours"]]
y = df["marks"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

print("\nModel Evaluation")
print("MAE:", round(mean_absolute_error(y_test, predictions), 2))
print("R2 Score:", round(r2_score(y_test, predictions), 2))

# Save model
joblib.dump(model, MODEL_PATH)

print("\nModel trained successfully!")
print("Model saved at:", MODEL_PATH)