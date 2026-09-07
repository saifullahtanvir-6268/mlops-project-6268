from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "housing.csv"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "housing_model.joblib"


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_pipeline() -> Pipeline:
    numeric_features = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
    ]
    categorical_features = ["ocean_proximity"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[SimpleImputer(strategy="median"), MinMaxScaler()]), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main() -> None:
    df = load_data(DATA_PATH)

    target = "median_house_value"
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    score = pipeline.score(X_test, y_test)
    print(f"R^2 on test set: {score:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
