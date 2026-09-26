
import json
import joblib
import mlflow
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from xgboost import XGBClassifier


MODEL_DIR = Path("tourism_project/deployment")
MODEL_PATH = MODEL_DIR / "best_model.joblib"
PARAM_PATH = MODEL_DIR / "best_params.json"
RESULT_PATH = MODEL_DIR / "model_results.json"


def load_data():
    """Load train and test datasets."""
    X_train = pd.read_csv("Xtrain.csv")
    X_test = pd.read_csv("Xtest.csv")
    y_train = pd.read_csv("ytrain.csv").squeeze()
    y_test = pd.read_csv("ytest.csv").squeeze()

    return X_train, X_test, y_train, y_test


def build_pipeline(X_train):
    """Create preprocessing and XGBoost pipeline."""
    categorical = X_train.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical = X_train.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numerical
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical
            )
        ]
    )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=2
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


def train_model():
    """Tune, evaluate, track and save the best model."""
    X_train, X_test, y_train, y_test = load_data()

    pipeline = build_pipeline(X_train)

    parameter_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.8, 1.0]
    }

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Tourism_Package_Prediction")

    with mlflow.start_run(run_name="XGBoost_GridSearch"):

        grid = GridSearchCV(
            pipeline,
            parameter_grid,
            scoring="roc_auc",
            cv=3,
            n_jobs=2,
            verbose=1
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        predictions = best_model.predict(X_test)
        probabilities = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1_score": f1_score(y_test, predictions),
            "roc_auc": roc_auc_score(y_test, probabilities)
        }

        mlflow.log_params(grid.best_params_)

        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(best_model, MODEL_PATH)

        with open(PARAM_PATH, "w") as file:
            json.dump(grid.best_params_, file, indent=4)

        with open(RESULT_PATH, "w") as file:
            json.dump(metrics, file, indent=4)

        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(PARAM_PATH))
        mlflow.log_artifact(str(RESULT_PATH))

        print("\nBEST PARAMETERS")
        print(json.dumps(grid.best_params_, indent=4))

        print("\nTEST METRICS")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        print("\nCLASSIFICATION REPORT")
        print(classification_report(y_test, predictions))

        print(f"\nModel saved: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
