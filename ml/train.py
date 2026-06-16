from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


FEATURES = ["seniority_num", "remote_score", "risk_num", "company_size_num", "job_title_enc"]


def train(df, output_dir="models"):
    if df is None or df.empty:
        raise ValueError("O DataFrame de treinamento está vazio.")

    df = df.copy()
    df["Job_Title"] = df["Job_Title"].fillna("Unknown").astype(str)

    label_encoder = LabelEncoder()
    df["job_title_enc"] = label_encoder.fit_transform(df["Job_Title"])

    for feature in FEATURES:
        if feature not in df.columns:
            df[feature] = np.nan

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = pd.to_numeric(df["Salary_USD"], errors="coerce")

    if y.dropna().empty:
        raise ValueError("Nenhuma coluna Salary_USD válida encontrada para treinamento.")

    median_salary = y.median()
    y = y.fillna(median_salary)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path / "salary_model.pkl")
    joblib.dump(label_encoder, output_path / "label_encoder.pkl")

    print(f"MAE:  {mae:,.0f}")
    print(f"RMSE: {rmse:,.0f}")
    print(f"R²:   {r2:.3f}")

    return model, {"mae": mae, "rmse": rmse, "r2": r2}
