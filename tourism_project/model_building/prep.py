
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_PATH = Path("tourism_project/data/tourism.csv")
TARGET = "ProdTaken"


def prepare_data():
    """Clean the dataset and create stratified train/test splits."""
    df = pd.read_csv(DATA_PATH)

    columns_to_remove = ["Unnamed: 0", "CustomerID"]
    df = df.drop(columns=columns_to_remove, errors="ignore")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    X_train.to_csv("Xtrain.csv", index=False)
    X_test.to_csv("Xtest.csv", index=False)
    y_train.to_csv("ytrain.csv", index=False)
    y_test.to_csv("ytest.csv", index=False)

    print("DATA PREPARATION SUCCESSFUL")
    print("=" * 40)
    print(f"Training rows : {len(X_train)}")
    print(f"Testing rows  : {len(X_test)}")
    print(f"Training target rate : {y_train.mean():.4f}")
    print(f"Testing target rate  : {y_test.mean():.4f}")


if __name__ == "__main__":
    prepare_data()
