from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# Project folder
project_folder = Path(__file__).resolve().parent.parent

# Dataset path
data_path = (
    project_folder
    / "data"
    / "processed"
    / "customer_churn_clean.csv"
)

# Load dataset
df = pd.read_csv(data_path)

# Separate features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Identify numerical and categorical columns
numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "category", "bool", "str"]
).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_columns),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns,
        ),
    ]
)

# Model pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ),
        ),
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Train model
model.fit(X_train, y_train)

# Extract feature names
feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

# Extract coefficients
coefficients = model.named_steps[
    "classifier"
].coef_[0]

# Create feature importance dataframe
importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": coefficients,
    }
)

# Sort by absolute importance
importance_df["Absolute_Importance"] = (
    importance_df["Importance"].abs()
)

importance_df = importance_df.sort_values(
    by="Absolute_Importance",
    ascending=False,
)

# Display top 15 features
top_features = importance_df.head(15)

print("\nTop 15 Features Affecting Customer Churn:")
print(
    top_features[
        ["Feature", "Importance"]
    ].to_string(index=False)
)

# Plot feature importance
plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"],
)

plt.xlabel("Logistic Regression Coefficient")
plt.ylabel("Features")

plt.title(
    "Top 15 Features Affecting Customer Churn"
)

plt.gca().invert_yaxis()

plt.tight_layout()

# Save plot
output_path = (
    project_folder
    / "feature_importance.png"
)

plt.savefig(output_path, dpi=300)

plt.show()

print(f"\nFeature importance chart saved to: {output_path}")