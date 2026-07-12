import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_excel("dataset/Flood dataset.xlsx")

print(data.columns)
print(data.info())
print(data.head())
data.columns=data.columns.str.strip()
# -------------------------------
# Descriptive Analysis
# -------------------------------
print("\n===== Dataset Shape =====")
print(data.shape)

print("\n===== Dataset Information =====")
print(data.info())

print("\n===== Descriptive Statistics =====")
print(data.describe())

print("\n===== Missing Values =====")
print(data.isnull().sum())
# -------------------------------
# Univariate Analysis
# -------------------------------
plt.figure(figsize=(6,4))
sns.histplot(data["Temp"], bins=10, kde=True)
plt.title("Temperature Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x=data["flood"])
plt.title("Flood Distribution")
plt.show()

# -------------------------------
# Multivariate Analysis
# -------------------------------
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# -------------------------------
# Outlier Analysis
# -------------------------------
plt.figure(figsize=(12,6))
sns.boxplot(data=data)
plt.xticks(rotation=45)
plt.title("Box Plot for Outlier Detection")
plt.tight_layout()
plt.show()
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
