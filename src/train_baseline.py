import argparse
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline


# ============================================================
# 경로
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)

REPORT_DIR = (
    BASE_DIR
    / "reports"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRAIN_PATH = (
    PROCESSED_DIR
    / "train.csv"
)

VALIDATION_PATH = (
    PROCESSED_DIR
    / "validation.csv"
)


# ============================================================
# Argument
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "TF-IDF + Logistic Regression "
            "Baseline Trainer"
        )
    )

    parser.add_argument(
        "--target",
        required=True,
        choices=[
            "type",
            "queue",
            "priority",
        ],
        help="학습할 target",
    )

    return parser.parse_args()


# ============================================================
# 데이터
# ============================================================

def load_data(
    target: str,
):

    print("=" * 70)
    print("1. 데이터 불러오기")
    print("=" * 70)

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    validation_df = pd.read_csv(
        VALIDATION_PATH
    )

    required_columns = [
        "text",
        target,
    ]

    for column in required_columns:

        if column not in train_df.columns:
            raise ValueError(
                f"Train 데이터에 "
                f"{column} 컬럼이 없습니다."
            )

        if column not in validation_df.columns:
            raise ValueError(
                f"Validation 데이터에 "
                f"{column} 컬럼이 없습니다."
            )

    train_df = train_df.dropna(
        subset=[
            "text",
            target,
        ]
    ).copy()

    validation_df = (
        validation_df.dropna(
            subset=[
                "text",
                target,
            ]
        )
        .copy()
    )

    train_df["text"] = (
        train_df["text"]
        .astype(str)
    )

    validation_df["text"] = (
        validation_df["text"]
        .astype(str)
    )

    print(
        f"Target     : {target}"
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
        f"클래스 수  : "
        f"{train_df[target].nunique()}"
    )

    print("\nClasses:")

    for value in sorted(
        train_df[target]
        .unique()
        .tolist()
    ):
        print(
            f"- {value}"
        )

    return (
        train_df,
        validation_df,
    )


# ============================================================
# 모델
# ============================================================

def build_model():

    print("\n")
    print("=" * 70)
    print("2. Baseline 모델 생성")
    print("=" * 70)

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    max_features=50_000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    print(
        "TF-IDF + Logistic Regression"
    )

    return pipeline


# ============================================================
# 학습
# ============================================================

def train_model(
    model,
    train_df,
    target: str,
):

    print("\n")
    print("=" * 70)
    print("3. 모델 학습")
    print("=" * 70)

    X_train = (
        train_df["text"]
    )

    y_train = (
        train_df[target]
    )

    print("학습 시작...")

    model.fit(
        X_train,
        y_train,
    )

    print("학습 완료.")

    return model


# ============================================================
# 평가
# ============================================================

def evaluate_model(
    model,
    validation_df,
    target: str,
):

    print("\n")
    print("=" * 70)
    print("4. Validation 평가")
    print("=" * 70)

    X_validation = (
        validation_df["text"]
    )

    y_validation = (
        validation_df[target]
    )

    predictions = model.predict(
        X_validation
    )

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    macro_f1 = f1_score(
        y_validation,
        predictions,
        average="macro",
    )

    weighted_f1 = f1_score(
        y_validation,
        predictions,
        average="weighted",
    )

    print(
        f"Accuracy    : {accuracy:.4f}"
    )

    print(
        f"Macro F1    : {macro_f1:.4f}"
    )

    print(
        f"Weighted F1 : {weighted_f1:.4f}"
    )

    print("\n")
    print(
        "Classification Report"
    )
    print("-" * 70)

    report = classification_report(
        y_validation,
        predictions,
        digits=4,
        zero_division=0,
    )

    print(report)

    labels = sorted(
        y_validation
        .unique()
        .tolist()
    )

    matrix = confusion_matrix(
        y_validation,
        predictions,
        labels=labels,
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=[
            f"actual_{label}"
            for label in labels
        ],
        columns=[
            f"pred_{label}"
            for label in labels
        ],
    )

    print("\n")
    print(
        "Confusion Matrix"
    )
    print("-" * 70)

    print(
        confusion_df
    )

    report_path = (
        REPORT_DIR
        / f"{target}_baseline_validation_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            f"Target: {target}\n\n"
        )

        file.write(
            f"Accuracy: {accuracy:.4f}\n"
        )

        file.write(
            f"Macro F1: {macro_f1:.4f}\n"
        )

        file.write(
            f"Weighted F1: "
            f"{weighted_f1:.4f}\n\n"
        )

        file.write(
            "Classification Report\n"
        )

        file.write(
            report
        )

        file.write(
            "\n\nConfusion Matrix\n"
        )

        file.write(
            confusion_df.to_string()
        )

    print(
        f"\n리포트 저장: "
        f"{report_path}"
    )


# ============================================================
# 저장
# ============================================================

def save_model(
    model,
    target: str,
):

    print("\n")
    print("=" * 70)
    print("5. 모델 저장")
    print("=" * 70)

    model_path = (
        MODEL_DIR
        / f"{target}_baseline.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        f"모델 저장 완료: "
        f"{model_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    target = args.target

    (
        train_df,
        validation_df,
    ) = load_data(
        target
    )

    model = build_model()

    model = train_model(
        model,
        train_df,
        target,
    )

    evaluate_model(
        model,
        validation_df,
        target,
    )

    save_model(
        model,
        target,
    )

    print("\n")
    print("=" * 70)
    print(
        f"{target.upper()} BASELINE "
        f"TRAINING COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()