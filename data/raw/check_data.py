from pathlib import Path
import pandas as pd

# Find the dataset folder
data_folder = Path(__file__).resolve().parent
# Find the CSV file
csv_files = list(data_folder.glob("*.csv"))

if not csv_files:
    print("No CSV file found in:", data_folder)
else:
    file_path = csv_files[0]
    print("Dataset found:", file_path.name)

    df = pd.read_csv(file_path)

    print("\nDataset loaded successfully!")
    print("Rows and columns:", df.shape)

    print("\nFirst five rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    if "Churn" in df.columns:
        print("\nCustomer churn counts:")
        print(df["Churn"].value_counts())
    else:
        print("\nThe Churn column was not found.")