from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

project_folder = Path(__file__).resolve().parent.parent
data_path = project_folder / "data" / "processed" / "customer_churn_clean.csv"

df = pd.read_csv(data_path)

X = df.drop(columns=["Churn"])
y = df["Churn"]

numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
categorical_columns = X.select_dtypes(
    include=["object", "category", "bool","str"]
).columns.tolist()

preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric_columns),
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )
}

results = []

for name, classifier in models.items():
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    from sklearn.metrics import precision_recall_curve

     # Find the best threshold using F1-score
    precision_values, recall_values, thresholds = precision_recall_curve(
    y_test, probabilities
)

    f1_values = (
    2 * precision_values[:-1] * recall_values[:-1]
    / (precision_values[:-1] + recall_values[:-1] + 1e-10)
)

    best_index = f1_values.argmax()
    best_threshold = thresholds[best_index]

    print("Best threshold:", round(best_threshold, 3))
    print("Best F1-score:", round(f1_values[best_index], 3))

    predictions = (probabilities >= best_threshold).astype(int)
    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, predictions), 3),
        "Precision": round(precision_score(y_test, predictions), 3),
        "Recall": round(recall_score(y_test, predictions), 3),
        "F1-score": round(f1_score(y_test, predictions), 3),
        "ROC-AUC": round(roc_auc_score(y_test, probabilities), 3),
    })

print("\nModel comparison:")
print(pd.DataFrame(results).to_string(index=False))

