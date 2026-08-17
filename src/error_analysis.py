from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

VALIDATION_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "validation.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "type_baseline.joblib"
)

REPORT_PATH = (
    BASE_DIR
    / "reports"
    / "problem_as_incident.csv"
)


def main():
    df = pd.read_csv(VALIDATION_PATH)

    model = joblib.load(MODEL_PATH)

    df["prediction"] = model.predict(
        df["text"]
    )

    errors = df[
        (df["type"] == "Problem")
        & (df["prediction"] == "Incident")
    ].copy()

    print(
        f"Problem → Incident 오분류: "
        f"{len(errors):,}건"
    )

    print("\n샘플 10개\n")

    for i, row in errors.head(10).iterrows():
        print("=" * 70)
        print(row["text"][:1000])
        print()

    errors.to_csv(
        REPORT_PATH,
        index=False,
    )

    print(
        f"\n오답 저장 완료: {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()