
import pandas as pd
from pathlib import Path

DATA_PATH = Path("tourism_project/data/tourism.csv")

EXPECTED_COLUMNS = [
    "Unnamed: 0",
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome"
]


def validate_dataset():
    """Validate the dataset schema and print a summary."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    missing = [
        column for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    print("DATA REGISTRATION SUCCESSFUL")
    print("=" * 40)
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")
    print(f"Missing : {df.isnull().sum().sum()}")

    print("\nExpected schema validated successfully.")

    print("\nTarget distribution:")
    print(df["ProdTaken"].value_counts())

    print("\nDataset summary:")
    print(df.describe(include="all").transpose())


if __name__ == "__main__":
    validate_dataset()
