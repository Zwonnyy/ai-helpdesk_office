from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# 프로젝트 경로
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
)

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def print_section(title: str) -> None:
    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)


def show_basic_info(df: pd.DataFrame) -> None:
    print_section("1. BASIC INFO")

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"- {column}")


def show_missing_values(df: pd.DataFrame) -> None:
    print_section("2. MISSING VALUES")

    missing = df.isna().sum()
    missing_rate = df.isna().mean() * 100

    result = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_rate(%)": missing_rate.round(2),
        }
    )

    print(result)


def show_duplicates(df: pd.DataFrame) -> None:
    print_section("3. DUPLICATES")

    total_duplicates = df.duplicated().sum()

    ticket_duplicates = df.duplicated(
        subset=["subject", "body"]
    ).sum()

    print(f"완전히 동일한 행: {total_duplicates:,}")
    print(f"subject + body 중복: {ticket_duplicates:,}")


def show_target_distribution(
    df: pd.DataFrame,
    column: str,
) -> None:
    print_section(f"TARGET DISTRIBUTION: {column}")

    counts = df[column].value_counts(dropna=False)
    ratios = df[column].value_counts(
        normalize=True,
        dropna=False,
    ) * 100

    result = pd.DataFrame(
        {
            "count": counts,
            "ratio(%)": ratios.round(2),
        }
    )

    print(result)

    counts.plot(
        kind="bar",
        title=f"{column} distribution",
    )

    plt.xlabel(column)
    plt.ylabel("count")
    plt.tight_layout()

    output_path = REPORT_DIR / f"{column}_distribution.png"

    plt.savefig(output_path)
    plt.close()

    print(f"\n그래프 저장: {output_path}")


def show_text_statistics(df: pd.DataFrame) -> None:
    print_section("4. TEXT LENGTH")

    subject = df["subject"].fillna("").astype(str)
    body = df["body"].fillna("").astype(str)

    text_stats = pd.DataFrame(
        {
            "subject_length": subject.str.len(),
            "body_length": body.str.len(),
            "total_length": (
                subject.str.len()
                + body.str.len()
            ),
        }
    )

    print(text_stats.describe().round(2))


def show_samples(df: pd.DataFrame) -> None:
    print_section("5. SAMPLE DATA")

    columns = [
        "subject",
        "type",
        "queue",
        "priority",
        "language",
    ]

    print(
        df[columns]
        .sample(10, random_state=42)
        .to_string(index=False)
    )


def main() -> None:
    print(f"Dataset: {DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"CSV 파일을 찾을 수 없습니다.\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "subject",
        "body",
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
            f"필수 컬럼이 없습니다: {missing_columns}"
        )

    show_basic_info(df)
    show_missing_values(df)
    show_duplicates(df)
    show_text_statistics(df)

    for column in [
        "type",
        "queue",
        "priority",
        "language",
    ]:
        show_target_distribution(df, column)

    show_samples(df)

    print_section("EDA COMPLETE")
    print("EDA가 정상적으로 완료되었습니다.")
    print(f"그래프 위치: {REPORT_DIR}")


if __name__ == "__main__":
    main()