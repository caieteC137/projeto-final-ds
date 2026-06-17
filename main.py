import pandas as pd

from etl.extract import extract_api, extract_csv, extract_pdf
from etl.load import load_to_json
from etl.transform import add_features, clean, integrate
from ml.train import train


def main():
    print("=== EXTRAÇÃO ===")
    df_csv = extract_csv()
    df_api = extract_api()

    try:
        df_pdf = extract_pdf()
    except FileNotFoundError as exc:
        print(f"PDF não encontrado: {exc}")
        df_pdf = pd.DataFrame(columns=["page", "text", "source"])

    print("=== TRANSFORMAÇÃO ===")
    df_csv = clean(df_csv, required_cols=["Salary_USD", "Job_Title"])
    df_csv = add_features(df_csv)

    df_api = clean(df_api, required_cols=["Job_Title"])
    df_api = add_features(df_api)

    df_final = integrate(df_csv, df_api, df_pdf)

    print("=== CARGA - ARQUIVO LOCAL ===")
    load_to_json(df_final)

    print("=== MACHINE LEARNING ===")
    train(df_final)

    print("=== CONCLUÍDO ===")
    print("Artefatos gerados:")
    print("  data/processed/dataset_final.csv")
    print("  data/processed/dataset_final.json")
    print("  models/salary_model.pkl")
    print("  models/label_encoder.pkl")


if __name__ == "__main__":
    main()
