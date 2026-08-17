from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# 경로
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

MODEL_DIR = (
    BASE_DIR
    / "models"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ANSWER_CORPUS_PATH = (
    PROCESSED_DIR
    / "answer_corpus.csv"
)

RETRIEVER_PATH = (
    MODEL_DIR
    / "answer_retriever.joblib"
)


# ============================================================
# 데이터 로딩
# ============================================================

def load_data() -> pd.DataFrame:

    print("=" * 70)
    print("1. ANSWER DATA LOAD")
    print("=" * 70)

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"원본 CSV가 없습니다.\n"
            f"{RAW_DATA_PATH}"
        )

    df = pd.read_csv(
        RAW_DATA_PATH
    )

    print(
        f"원본 데이터: {len(df):,}건"
    )

    return df


# ============================================================
# 검색용 데이터 정제
# ============================================================

def clean_data(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print("\n")
    print("=" * 70)
    print("2. ANSWER CORPUS CLEANING")
    print("=" * 70)

    required_columns = [
        "subject",
        "body",
        "answer",
        "type",
        "queue",
        "priority",
        "language",
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

    df = df[
        required_columns
    ].copy()

    # --------------------------------------------------------
    # 문자열 정리
    # --------------------------------------------------------

    for column in [
        "subject",
        "body",
        "answer",
    ]:
        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    # --------------------------------------------------------
    # 검색용 text 생성
    # --------------------------------------------------------

    df["text"] = (
        df["subject"]
        + "\n"
        + df["body"]
    ).str.strip()

    # 문의가 비어있거나 답변이 없는 데이터 제거
    before = len(df)

    df = df[
        (df["text"].str.len() > 0)
        & (df["answer"].str.len() > 0)
    ].copy()

    print(
        "빈 문의/답변 제거: "
        f"{before - len(df):,}건"
    )

    # --------------------------------------------------------
    # 중복 제거
    # --------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(
        subset=[
            "text",
            "answer",
        ]
    )

    print(
        "중복 제거: "
        f"{before - len(df):,}건"
    )

    df = df[
        [
            "subject",
            "body",
            "text",
            "answer",
            "type",
            "queue",
            "priority",
            "language",
        ]
    ].reset_index(
        drop=True
    )

    print(
        f"최종 검색 Corpus: "
        f"{len(df):,}건"
    )

    return df


# ============================================================
# TF-IDF 검색 인덱스 생성
# ============================================================

def build_retriever(
    df: pd.DataFrame,
):

    print("\n")
    print("=" * 70)
    print("3. TF-IDF RETRIEVER BUILD")
    print("=" * 70)

    vectorizer = TfidfVectorizer(
        lowercase=True,

        ngram_range=(1, 2),

        min_df=2,

        max_df=0.95,

        max_features=100_000,

        sublinear_tf=True,

        norm="l2",
    )

    print(
        "TF-IDF 학습 시작..."
    )

    matrix = (
        vectorizer.fit_transform(
            df["text"]
        )
    )

    print(
        "TF-IDF 학습 완료."
    )

    print(
        f"Matrix shape: "
        f"{matrix.shape}"
    )

    print(
        f"Vocabulary: "
        f"{len(vectorizer.vocabulary_):,}"
    )

    return (
        vectorizer,
        matrix,
    )


# ============================================================
# 저장
# ============================================================

def save_retriever(
    df,
    vectorizer,
    matrix,
):

    print("\n")
    print("=" * 70)
    print("4. SAVE RETRIEVER")
    print("=" * 70)

    df.to_csv(
        ANSWER_CORPUS_PATH,
        index=False,
    )

    bundle = {
        "vectorizer": vectorizer,
        "matrix": matrix,
    }

    joblib.dump(
        bundle,
        RETRIEVER_PATH,
        compress=3,
    )

    print(
        f"Corpus 저장:"
        f"\n{ANSWER_CORPUS_PATH}"
    )

    print(
        f"\nRetriever 저장:"
        f"\n{RETRIEVER_PATH}"
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
        vectorizer,
        matrix,
    ) = build_retriever(
        df
    )

    save_retriever(
        df,
        vectorizer,
        matrix,
    )

    print("\n")
    print("=" * 70)
    print(
        "ANSWER RETRIEVER READY"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()