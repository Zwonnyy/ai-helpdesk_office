from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[1]

SOURCE_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "rag_validation.csv"
)

DEV_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "v3_dev.csv"
)

FINAL_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "final_holdout.csv"
)


def main():

    df = pd.read_csv(
        SOURCE_PATH
    )

    dev_df, final_df = train_test_split(
        df,
        test_size=0.5,
        random_state=2026,
        stratify=df["type"],
    )

    dev_df = dev_df.reset_index(
        drop=True
    )

    final_df = final_df.reset_index(
        drop=True
    )

    dev_df.to_csv(
        DEV_PATH,
        index=False,
    )

    final_df.to_csv(
        FINAL_PATH,
        index=False,
    )

    print("=" * 70)
    print("V3 EVALUATION SPLIT READY")
    print("=" * 70)

    print(
        f"Development    : {len(dev_df):,}"
    )

    print(
        f"Final Holdout  : {len(final_df):,}"
    )

    print()
    print(
        "V3/V4 튜닝에는 v3_dev.csv만 사용합니다."
    )

    print(
        "final_holdout.csv는 최종 모델 결정 전까지 사용하지 않습니다."
    )


if __name__ == "__main__":
    main()