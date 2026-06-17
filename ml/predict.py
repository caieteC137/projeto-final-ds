from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "salary_model.pkl"
ENCODER_PATH = Path(__file__).resolve().parents[1] / "models" / "label_encoder.pkl"


try:
    from etl.transform import normalize_title
except ImportError:
    def normalize_title(title):
        return str(title).strip().title()


def _load_assets():
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    return model, label_encoder


def _normalise_remote_ratio(remote_ratio):
    value = float(remote_ratio)
    if value > 1:
        value = value / 100
    return value


def predict_salary(
    job_title,
    seniority,
    remote_ratio,
    automation_risk,
    company_size="unknown",
):
    model, label_encoder = _load_assets()

    seniority_map = {
        "junior": 0,
        "entry": 0,
        "en": 0,
        "mid": 1,
        "middle": 1,
        "mi": 1,
        "senior": 2,
        "se": 2,
        "lead": 3,
        "staff": 3,
        "principal": 3,
        "ex": 3,
    }

    risk_map = {
        "low": 0,
        "medium": 1,
        "mid": 1,
        "high": 2,
    }

    company_size_map = {
        "small": 0,
        "medium": 1,
        "large": 2,
        "enterprise": 2,
    }

    title = normalize_title(job_title)
    title_encoded = label_encoder.transform([title])[0] if title in label_encoder.classes_ else 0

    X = pd.DataFrame(
        [
            {
                "seniority_num": seniority_map.get(str(seniority).lower(), 1),
                "remote_score": _normalise_remote_ratio(remote_ratio),
                "risk_num": risk_map.get(str(automation_risk).lower(), 1),
                "company_size_num": company_size_map.get(str(company_size).lower(), np.nan),
                "job_title_enc": title_encoded,
            }
        ]
    )

    return float(model.predict(X)[0])
