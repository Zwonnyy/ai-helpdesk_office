from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
)

PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_and_clean():

    df = pd.read_csv(
        RAW_DATA_PATH
    )

    columns = [
        "subject",
        "body",
        "answer",
        "type",
        "queue",
        "priority",
        "language",
    ]

    df = df[
        columns
    ].copy()

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

    df["text"] = (
        df["subject"]
        + "\n"
        + df["body"]
    ).str.strip()

    df = df[
        (df["text"].str.len() > 0)
        & (df["answer"].str.len() > 0)
        & (df["type"].notna())
    ].copy()

    df = df.drop_duplicates(
        subset=[
            "text",
            "answer",
        ]
    ).reset_index(
        drop=True
    )

    return df


def split_data(
    df,
):

    train_df, temp_df = (
        train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df["type"],
        )
    )

    validation_df, test_df = (
        train_test_split(
            temp_df,
            test_size=0.5,
            random_state=42,
            stratify=temp_df["type"],
        )
    )

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_retriever(
    train_df,
):

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=100_000,
        sublinear_tf=True,
        norm="l2",
    )

    matrix = (
        vectorizer.fit_transform(
            train_df["text"]
        )
    )

    return (
        vectorizer,
        matrix,
    )


def main():

    print("=" * 70)
    print("RAG DATA PREPARATION")
    print("=" * 70)

    df = load_and_clean()

    print(
        f"전체 데이터: {len(df):,}"
    )

    train_df, validation_df, test_df = (
        split_data(df)
    )

    print(
        f"Train      : {len(train_df):,}"
    )

    print(
        f"Validation : {len(validation_df):,}"
    )

    print(
        f"Test       : {len(test_df):,}"
    )

    train_df.to_csv(
        PROCESSED_DIR
        / "rag_train.csv",
        index=False,
    )

    validation_df.to_csv(
        PROCESSED_DIR
        / "rag_validation.csv",
        index=False,
    )

    test_df.to_csv(
        PROCESSED_DIR
        / "rag_test.csv",
        index=False,
    )

    vectorizer, matrix = (
        build_retriever(
            train_df
        )
    )

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "matrix": matrix,
        },
        MODEL_DIR
        / "answer_retriever.joblib",
    )

    # FastAPI가 기존 파일명을 사용하므로
    train_df.to_csv(
        PROCESSED_DIR
        / "answer_corpus.csv",
        index=False,
    )

    print()
    print(
        "RAG Knowledge Base 생성 완료"
    )

    print(
        f"Matrix: {matrix.shape}"
    )


if __name__ == "__main__":
    main()