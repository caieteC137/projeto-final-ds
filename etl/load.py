import json
from pathlib import Path

import numpy as np
import pandas as pd


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return str(value)


def _clean_record(record):
    cleaned = {}
    for key, value in record.items():
        if isinstance(value, np.ndarray):
            cleaned[key] = [item for item in value.tolist() if not pd.isna(item)]
        elif isinstance(value, (list, tuple, set)):
            cleaned[key] = [item for item in value if not pd.isna(item)]
        elif isinstance(value, dict):
            cleaned[key] = _clean_record(value)
        elif pd.isna(value):
            cleaned[key] = None
        else:
            cleaned[key] = value
    return cleaned


def load_to_json(df, output_path="data/processed/dataset_final.json"):
    if df is None or df.empty:
        raise ValueError("Nenhum dado disponível para salvar no arquivo local.")

    records = [_clean_record(record) for record in df.to_dict("records")]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    print(f"{len(records)} registros salvos em {output}.")
    return len(records)
