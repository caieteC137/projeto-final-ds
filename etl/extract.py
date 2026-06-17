import os
import pandas as pd
import requests
import pdfplumber


EXPECTED_API_COLUMNS = [
    "title",
    "company_name",
    "category",
    "candidate_required_location",
    "salary",
    "tags",
    "source",
]


def extract_csv(path="data/raw/kaggle_salaries.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV não encontrado: {path}")

    df = pd.read_csv(path)
    df["source"] = "kaggle"
    return df


def extract_api(url="https://remotive.com/api/remote-jobs?category=software-dev&limit=100"):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    jobs = payload.get("jobs", [])

    df = pd.DataFrame(jobs)
    if df.empty:
        return pd.DataFrame(columns=EXPECTED_API_COLUMNS)

    for column in EXPECTED_API_COLUMNS:
        if column not in df.columns:
            df[column] = None

    df = df[EXPECTED_API_COLUMNS]
    df["source"] = "remotive"
    return df


def extract_pdf(path="data/pdf/wef_future_of_jobs_2025.pdf"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF não encontrado: {path}")

    records = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                records.append({"page": i + 1, "text": text, "source": "wef_pdf"})

    return pd.DataFrame(records)
