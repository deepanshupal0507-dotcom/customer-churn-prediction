from pathlib import Path
import pandas as pd

# Load the dataset
project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data" / "raw"

csv_files = list(data_folder.glob("*.csv"))

if not csv_files:
    print("Dataset not found. Check the data/raw folder.")
else:
    df = pd.read_csv(csv_files[0])

    # Basic dataset information
    print("Dataset shape:", df.shape)

    print("\nData types:")
    print(df.dtypes)

    # Missing values
    print("\nMissing values by column:")
    print(df.isnull().sum())

    # Duplicate records
    print("\nDuplicate rows:", df.duplicated().sum())

    # Churn distribution
    print("\nChurn counts:")
    print(df["Churn"].value_counts())

    print("\nChurn percentage:")
    print(
        (df["Churn"].value_counts(normalize=True) * 100).round(2)
    )
    print("\nChecking TotalCharges...")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("Missing TotalCharges:", df["TotalCharges"].isnull().sum())

print("\nCustomer tenure summary:")
print(df["tenure"].describe())

print("\nContract types:")
print(df["Contract"].value_counts())

print("\nChurn by contract:")
print(pd.crosstab(df["Contract"], df["Churn"], normalize="index").round(3) * 100)