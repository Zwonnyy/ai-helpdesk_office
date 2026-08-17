from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 경로 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
)

PROCESSED_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_STATE = 42


# ============================================================
# 데이터 불러오기
# ============================================================

def load_data() -> pd.DataFrame:
    print("=" * 70)
    print("1. 원본 데이터 불러오기")
    print("=" * 70)

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"원본 CSV 파일을 찾을 수 없습니다.\n"
            f"{RAW_DATA_PATH}"
        )

    df = pd.read_csv(
        RAW_DATA_PATH
    )

    print(
        f"원본 데이터: "
        f"{len(df):,}건"
    )

    print(
        f"컬럼 수: "
        f"{len(df.columns)}개"
    )

    return df


# ============================================================
# 데이터 정제
# ============================================================

def clean_data(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print("\n")
    print("=" * 70)
    print("2. 데이터 정제")
    print("=" * 70)

    required_columns = [
        "subject",
        "body",
        "type",
        "queue",
        "priority",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"필수 컬럼이 없습니다: "
            f"{missing_columns}"
        )

    # 우리가 사용할 컬럼만 복사
    df = df[
        required_columns
    ].copy()

    print(
        f"정제 전 데이터: "
        f"{len(df):,}건"
    )

    # --------------------------------------------------------
    # Target 결측 제거
    # --------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=[
            "type",
            "queue",
            "priority",
        ]
    )

    print(
        "Target 결측 제거: "
        f"{before - len(df):,}건"
    )

    # --------------------------------------------------------
    # 텍스트 결측 처리
    # --------------------------------------------------------

    df["subject"] = (
        df["subject"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["body"] = (
        df["body"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # 제목 + 본문 합치기
    # --------------------------------------------------------

    df["text"] = (
        df["subject"]
        + "\n"
        + df["body"]
    ).str.strip()

    # 빈 문자열 제거
    before = len(df)

    df = df[
        df["text"].str.len() > 0
    ].copy()

    print(
        "빈 텍스트 제거: "
        f"{before - len(df):,}건"
    )

    # --------------------------------------------------------
    # 중복 제거
    # --------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(
        subset=[
            "text",
            "type",
            "queue",
            "priority",
        ]
    )

    print(
        "중복 제거: "
        f"{before - len(df):,}건"
    )

    # --------------------------------------------------------
    # 최종 컬럼
    # --------------------------------------------------------

    df = df[
        [
            "text",
            "type",
            "queue",
            "priority",
        ]
    ].reset_index(
        drop=True
    )

    print(
        f"정제 후 데이터: "
        f"{len(df):,}건"
    )

    return df


# ============================================================
# Train / Validation / Test 분할
# ============================================================

def split_data(
    df: pd.DataFrame,
):

    print("\n")
    print("=" * 70)
    print("3. Train / Validation / Test 분할")
    print("=" * 70)

    # 기존 Type 실험과 최대한 같은 조건을 유지하기 위해
    # type 기준 stratified split 사용

    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["type"],
    )

    validation_df, test_df = (
        train_test_split(
            temp_df,
            test_size=0.50,
            random_state=RANDOM_STATE,
            stratify=temp_df["type"],
        )
    )

    print(
        f"Train      : "
        f"{len(train_df):,}"
    )

    print(
        f"Validation : "
        f"{len(validation_df):,}"
    )

    print(
        f"Test       : "
        f"{len(test_df):,}"
    )

    return (
        train_df,
        validation_df,
        test_df,
    )


# ============================================================
# 분포 확인
# ============================================================

def show_distribution(
    name: str,
    df: pd.DataFrame,
) -> None:

    print("\n")
    print("-" * 70)
    print(
        f"{name} DISTRIBUTION"
    )
    print("-" * 70)

    for target in [
        "type",
        "queue",
        "priority",
    ]:

        print(
            f"\n[{target}]"
        )

        distribution = (
            df[target]
            .value_counts(
                normalize=True
            )
            .mul(100)
            .round(2)
        )

        print(distribution)


# ============================================================
# 저장
# ============================================================

def save_data(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:

    print("\n")
    print("=" * 70)
    print("4. 데이터 저장")
    print("=" * 70)

    train_path = (
        PROCESSED_DIR
        / "train.csv"
    )

    validation_path = (
        PROCESSED_DIR
        / "validation.csv"
    )

    test_path = (
        PROCESSED_DIR
        / "test.csv"
    )

    train_df.to_csv(
        train_path,
        index=False,
    )

    validation_df.to_csv(
        validation_path,
        index=False,
    )

    test_df.to_csv(
        test_path,
        index=False,
    )

    print(
        f"Train      → {train_path}"
    )

    print(
        f"Validation → {validation_path}"
    )

    print(
        f"Test       → {test_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    df = load_data()

    df = clean_data(
        df
    )

    (
        train_df,
        validation_df,
        test_df,
    ) = split_data(
        df
    )

    show_distribution(
        "TRAIN",
        train_df,
    )

    show_distribution(
        "VALIDATION",
        validation_df,
    )

    show_distribution(
        "TEST",
        test_df,
    )

    save_data(
        train_df,
        validation_df,
        test_df,
    )

    print("\n")
    print("=" * 70)
    print("PREPROCESS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()