import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load dataset
data = pd.read_excel("dataset/Flood dataset.xlsx")

print(data.columns)
data.columns=data.columns.str.strip()

# Input and Output
X = data.drop("flood", axis=1)
y = data["flood"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/flood_model.pkl")

print("Model trained successfully!")