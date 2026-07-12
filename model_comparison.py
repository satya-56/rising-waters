import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_excel("dataset/flood dataset.xlsx")   # Dataset path ni mee project prakaram marchandi

print(df.columns)
print(df.head())

# Features and Target
X = df.drop("flood", axis=1)   # "Flood" ni mee target column name tho replace cheyyandi
y = df["flood"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier()
}

# Compare models
print("Model Comparison Results:\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{name}: {accuracy * 100:.2f}% Accuracy")

best_model = RandomForestClassifier(random_state=42)
best_model.fit(X_train, y_train)

joblib.dump(best_model, "models/best_model.pkl")

print("Best model saved successfully!")
