from pathlib import Path

import pandas as pd
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# Locate the cleaned dataset
project_folder = Path(__file__).resolve().parent.parent
data_path = (
    project_folder
    / "data"
    / "processed"
    / "customer_churn_clean.csv"
)

df = pd.read_csv(data_path)

# Separate features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Identify numeric and categorical columns
numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# Preprocess the two types of features
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_columns),
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_columns),
    ]
)

# Split data before fitting preprocessing or the model
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Build the model pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

# Train and evaluate
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("Model training completed!")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))

print("\nClassification report:")
print(classification_report(y_test, predictions))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))
model_folder = project_folder / "models"
model_folder.mkdir(parents=True, exist_ok=True)

model_path = model_folder / "logistic_regression_model.pkl"

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print(f"Model location: {model_path}")