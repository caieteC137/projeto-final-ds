import hashlib
import json
import re
import numpy as np
import pandas as pd


COMMON_COLUMNS = [
    "source",
    "Job_Title",
    "Salary_USD",
    "Location",
    "Location_Country",
    "Experience_Level",
    "Remote_Ratio",
    "Company_Size",
    "Required_Skills",
    "Automation_Risk",
    "seniority_num",
    "remote_score",
    "risk_num",
    "company_size_num",
    "skills_list",
    "row_hash",
]


COUNTRY_MAP = {
    "united states": "US",
    "usa": "US",
    "us": "US",
    "america": "US",
    "brazil": "BR",
    "brasil": "BR",
    "br": "BR",
    "united kingdom": "GB",
    "uk": "GB",
    "england": "GB",
    "canada": "CA",
    "germany": "DE",
    "france": "FR",
    "spain": "ES",
    "portugal": "PT",
    "italy": "IT",
    "netherlands": "NL",
    "australia": "AU",
    "india": "IN",
    "mexico": "MX",
    "argentina": "AR",
    "chile": "CL",
    "colombia": "CO",
    "peru": "PE",
    "south africa": "ZA",
    "japan": "JP",
    "china": "CN",
    "singapore": "SG",
    "ireland": "IE",
    "switzerland": "CH",
    "sweden": "SE",
    "poland": "PL",
    "worldwide": "WW",
    "global": "WW",
    "remote": "WW",
    "anywhere": "WW",
}

TITLE_MAP = {
    "machine learning engineer": "ML Engineer",
    "ml engineer": "ML Engineer",
    "artificial intelligence engineer": "AI Engineer",
    "ai engineer": "AI Engineer",
    "data scientist": "Data Scientist",
    "data science": "Data Scientist",
    "data engineer": "Data Engineer",
    "software engineer": "Software Engineer",
    "backend engineer": "Backend Engineer",
    "front end engineer": "Frontend Engineer",
    "frontend engineer": "Frontend Engineer",
    "full stack engineer": "Full Stack Engineer",
    "fullstack engineer": "Full Stack Engineer",
    "full stack developer": "Full Stack Engineer",
    "devops engineer": "DevOps Engineer",
    "mlops engineer": "MLOps Engineer",
    "nlp engineer": "NLP Engineer",
    "computer vision engineer": "Computer Vision Engineer",
    "research scientist": "Research Scientist",
    "ai research scientist": "Research Scientist",
    "cloud engineer": "Cloud Engineer",
    "product manager": "Product Manager",
    "business analyst": "Business Analyst",
    "qa engineer": "QA Engineer",
    "quality assurance": "QA Engineer",
    "cybersecurity analyst": "Cybersecurity Analyst",
    "security analyst": "Cybersecurity Analyst",
    "intern": "Intern",
    "trainee": "Trainee",
}

EXPERIENCE_MAP = {
    "EN": 0,
    "MI": 1,
    "SE": 2,
    "EX": 3,
}

COMPANY_SIZE_MAP = {
    "Small": 0,
    "Medium": 1,
    "Large": 2,
    "Unknown": np.nan,
}


def _record_hash(row):
    serializable = row.dropna().astype(str).to_dict()
    payload = json.dumps(serializable, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _make_hashable_for_dedupe(df):
    dedupe_df = df.copy()
    for column in dedupe_df.select_dtypes(include="object").columns:
        if dedupe_df[column].map(lambda value: isinstance(value, (list, tuple, dict, set))).any():
            dedupe_df[column] = dedupe_df[column].apply(
                lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
                if isinstance(value, (list, tuple, dict, set))
                else value
            )
    return dedupe_df


def clean(df, required_cols):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    for column in required_cols:
        if column not in df.columns:
            df[column] = np.nan

    df = df.dropna(subset=required_cols)
    dedupe_df = _make_hashable_for_dedupe(df)
    df = df.loc[~dedupe_df.duplicated(keep="first")].copy()
    df["row_hash"] = df.apply(_record_hash, axis=1)
    df = df.drop_duplicates(subset=["row_hash"])
    df = df.drop(columns=["row_hash"])
    return df.reset_index(drop=True)


def parse_salary(salary_str):
    if pd.isna(salary_str):
        return np.nan

    text = str(salary_str).strip().lower()
    if text in {"", "nan", "none", "null"}:
        return np.nan

    multiplier = 1
    if any(term in text for term in ["hour", "hora", "/h"]):
        multiplier = 2080
    elif any(term in text for term in ["month", "mensal", "/mo", "/mês", "/mes"]):
        multiplier = 12

    text = re.sub(r"million", "000000", text)
    text = re.sub(r"thousand", "000", text)
    text = re.sub(r"(?<=\d)\s*k\b", "000", text)
    text = re.sub(r"(?<=\d)\s*m\b", "000000", text)
    text = re.sub(r"[^\d.,\-/]", " ", text)
    text = text.replace(" ", "")

    if "," in text and "." not in text:
        text = text.replace(",", "")
    else:
        text = text.replace(",", ".")

    values = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", text)]
    if not values:
        return np.nan

    salary = float(np.mean(values)) * multiplier
    return round(salary, 2)


def normalize_country(value):
    if pd.isna(value):
        return "UNK"

    key = re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()
    if not key:
        return "UNK"
    if key in COUNTRY_MAP:
        return COUNTRY_MAP[key]
    for alias, code in COUNTRY_MAP.items():
        if alias and alias in key:
            return code
    if len(key) == 2 and key.isalpha():
        return key.upper()
    return key.replace(" ", "-").upper()[:10]


def normalize_title(title):
    if pd.isna(title):
        return "Unknown"

    normalized = re.sub(r"[^a-z0-9]+", " ", str(title).strip().lower()).strip()
    if not normalized:
        return "Unknown"

    for key, value in TITLE_MAP.items():
        if key in normalized:
            return value

    return " ".join(word.capitalize() for word in normalized.split())


def normalize_experience_level(value):
    if pd.isna(value):
        return "Unknown"

    text = str(value).strip().lower()
    text = text.replace("ê", "e").replace("é", "e").replace("ã", "a").replace("á", "a")
    text = re.sub(r"[^a-z0-9]+", " ", text).strip().upper()
    if not text:
        return "Unknown"
    if text in {"EN", "MI", "SE", "EX"}:
        return text
    if any(token in text for token in ["ENTRY", "JUNIOR", "JR", "INTERN", "TRAINEE"]):
        return "EN"
    if any(token in text for token in ["MID", "MIDDLE", "PLENO"]):
        return "MI"
    if any(token in text for token in ["SENIOR", "SR"]):
        return "SE"
    if any(token in text for token in ["LEAD", "STAFF", "PRINCIPAL", "EXECUTIVE", "ESPECIALISTA"]):
        return "EX"
    return "Unknown"


def normalize_company_size(value):
    if pd.isna(value):
        return "Unknown"

    text = re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()
    if "small" in text or "pequ" in text:
        return "Small"
    if "medium" in text or "medio" in text or "médio" in text:
        return "Medium"
    if "large" in text or "grande" in text or "enterprise" in text:
        return "Large"
    return "Unknown"


def _seniority_from_title(title):
    text = str(title).lower()
    if any(token in text for token in ["junior", "jr", "intern", "trainee"]):
        return 0
    if any(token in text for token in ["mid", "middle", "pleno"]):
        return 1
    if any(token in text for token in ["lead", "staff", "principal", "especialista"]):
        return 3
    if any(token in text for token in ["senior", "sênior", "sr"]):
        return 2
    return np.nan


def _parse_remote_score(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.number)):
        score = float(value)
        if score > 1:
            score = score / 100
        return round(float(np.clip(score, 0, 1)), 3)

    text = str(value).strip().lower()
    if not text:
        return np.nan
    if any(term in text for term in ["remote", "remoto", "100"]):
        return 1.0
    if any(term in text for term in ["hybrid", "híbrido", "hibrido", "50"]):
        return 0.5
    if any(term in text for term in ["onsite", "presencial", "office"]):
        return 0.0

    values = re.findall(r"\d+(?:\.\d+)?", text)
    if values:
        score = float(values[0])
        if score > 1:
            score = score / 100
        return round(float(np.clip(score, 0, 1)), 3)

    return np.nan


def _parse_risk_num(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()
    if not text:
        return np.nan
    if any(term in text for term in ["high", "alto", "elevado"]):
        return 2
    if any(term in text for term in ["medium", "mid", "médio", "medio", "moderate"]):
        return 1
    if any(term in text for term in ["low", "baixo"]):
        return 0
    return np.nan


def _parse_skills(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if pd.isna(value):
        return []

    text = str(value).strip().strip("[]")
    parts = re.split(r"[,;|]", text)
    return [part.strip().strip("\"'") for part in parts if part.strip()]


def add_features(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=COMMON_COLUMNS)

    df = df.copy()
    for column in [
        "Job_Title",
        "Salary_USD",
        "Location",
        "Experience_Level",
        "Remote_Ratio",
        "Company_Size",
        "Required_Skills",
        "Automation_Risk",
        "source",
    ]:
        if column not in df.columns:
            df[column] = np.nan

    df["source"] = df["source"].fillna("unknown")
    df["Job_Title"] = df["Job_Title"].apply(normalize_title)
    df["Salary_USD"] = df["Salary_USD"].apply(parse_salary)
    df["Location_Country"] = df["Location"].apply(normalize_country)
    df["Experience_Level"] = df["Experience_Level"].apply(normalize_experience_level)
    df["Company_Size"] = df["Company_Size"].apply(normalize_company_size)
    df["company_size_num"] = df["Company_Size"].map(COMPANY_SIZE_MAP)
    df["seniority_num"] = df.apply(
        lambda row: EXPERIENCE_MAP.get(row.get("Experience_Level"), _seniority_from_title(row.get("Job_Title"))),
        axis=1,
    )
    df["remote_score"] = df["Remote_Ratio"].apply(_parse_remote_score)
    df["risk_num"] = df["Automation_Risk"].apply(_parse_risk_num)
    df["skills_list"] = df["Required_Skills"].apply(_parse_skills)

    for column in ["seniority_num", "remote_score", "risk_num", "company_size_num"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def _prepare_kaggle(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=COMMON_COLUMNS)

    prepared = df.copy()
    prepared["source"] = "kaggle"
    return add_features(prepared)


def _prepare_api(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=COMMON_COLUMNS)

    prepared = pd.DataFrame()
    prepared["source"] = "remotive"
    prepared["Job_Title"] = df.get("title", pd.Series(dtype=object))
    prepared["Company_Name"] = df.get("company_name", pd.Series(dtype=object))
    prepared["Category"] = df.get("category", pd.Series(dtype=object))
    prepared["Location"] = df.get("candidate_required_location", pd.Series(dtype=object))
    prepared["Salary_USD"] = df.get("salary", pd.Series(dtype=object)).apply(parse_salary)
    prepared["Required_Skills"] = df.get("tags", pd.Series(dtype=object))
    prepared["Remote_Ratio"] = 100
    prepared["Experience_Level"] = "Unknown"
    prepared["Company_Size"] = "Unknown"
    prepared["Automation_Risk"] = "Medium"
    return add_features(prepared)


def _prepare_pdf(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=COMMON_COLUMNS)

    prepared = df.copy()
    if "job_title" in prepared.columns and "Job_Title" not in prepared.columns:
        prepared = prepared.rename(columns={"job_title": "Job_Title"})
    if "salary_usd" in prepared.columns and "Salary_USD" not in prepared.columns:
        prepared = prepared.rename(columns={"salary_usd": "Salary_USD"})
    if "salary" in prepared.columns and "Salary_USD" not in prepared.columns:
        prepared["Salary_USD"] = prepared["salary"].apply(parse_salary)
    if "location" not in prepared.columns:
        prepared["Location"] = "WW"
    if "source" not in prepared.columns:
        prepared["source"] = "wef_pdf"

    required = ["Job_Title", "Salary_USD"]
    if any(column not in prepared.columns for column in required):
        return pd.DataFrame(columns=COMMON_COLUMNS)

    return add_features(prepared)


def integrate(df_kaggle, df_api, df_pdf_summary=None, output_path="data/processed/dataset_final.csv"):
    frames = [
        _prepare_kaggle(df_kaggle),
        _prepare_api(df_api),
        _prepare_pdf(df_pdf_summary),
    ]
    combined = pd.concat([frame for frame in frames if frame is not None and not frame.empty], ignore_index=True)

    if combined.empty:
        combined = pd.DataFrame(columns=COMMON_COLUMNS)
        combined.to_csv(output_path, index=False)
        return combined

    combined = clean(combined, required_cols=["Salary_USD", "Job_Title"])
    combined = combined.drop_duplicates(subset=["source", "Job_Title", "Location", "Salary_USD"])
    combined["row_hash"] = combined.apply(_record_hash, axis=1)
    combined = combined[COMMON_COLUMNS]
    combined.to_csv(output_path, index=False)
    return combined
