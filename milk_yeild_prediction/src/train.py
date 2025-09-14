import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from src.data_loader import load_data
from src.feature_engineering import feature_engineering
from src.outlier import cap_outliers
from src.evaluation import evaluate
from src.preprocessing import get_preprocessor
from src.models import get_stacked_model

import joblib
import os


def main():
    # 1. Load Data
    df = load_data("data/synthetic_milk_yield_dataset.csv")

    # 2. Feature Engineering
    df = feature_engineering(df)

    # 3. Prepare Target and Features
    target = "cow_mean_yield"
    X = df.drop(columns=[target], errors="ignore")
    y = df[target]

    # 4. Outlier Capping
    num_cols = X.select_dtypes(include=[np.number]).columns
    X = cap_outliers(X, num_cols)

    # 5. Preprocessing
    preprocessor, numeric_features, categorical_features = get_preprocessor(X)

    # 6. Model
    stacked_model = get_stacked_model()
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", stacked_model)
    ])

    # 7. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 8. Fit Model
    pipeline.fit(X_train, y_train)

    # 9. Predictions
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # 10. Evaluation
    print("===== Hold-out Evaluation =====")
    evaluate(y_train, y_train_pred, "Train")
    evaluate(y_test, y_test_pred, "Test")

    # 11. Cross-validation
    print("\n===== 5-Fold Cross-Validation =====")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2", n_jobs=-1)
    print(f"CV R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f}")

    # 12. Save the trained pipeline
    os.makedirs("model", exist_ok=True)
    joblib.dump(pipeline, "model/milk_yield_pipeline.pkl")
    print("\nModel saved to model/milk_yield_pipeline.pkl")

if __name__ == "__main__":
    main()
