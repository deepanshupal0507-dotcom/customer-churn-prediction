from pathlib import Path
import pandas as pd

# Locate and load the dataset
project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data" / "raw"
csv_files = list(data_folder.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("Dataset not found in data/raw")

df = pd.read_csv(csv_files[0])

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Check missing values before cleaning
print("Missing TotalCharges before cleaning:",
      df["TotalCharges"].isna().sum())

# Remove rows with missing TotalCharges
df = df.dropna(subset=["TotalCharges"]).copy()

# Remove customerID because it is an identifier
df = df.drop(columns=["customerID"])

# Convert target to numeric
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# Save the cleaned dataset
processed_folder = project_folder / "data" / "processed"
processed_folder.mkdir(parents=True, exist_ok=True)

output_path = processed_folder / "customer_churn_clean.csv"
df.to_csv(output_path, index=False)

print("\nData cleaning completed.")
print("Final dataset shape:", df.shape)
print("Remaining missing values:", df.isna().sum().sum())
print("Churn distribution:")
print(df["Churn"].value_counts())
print("\nSaved to:", output_path)