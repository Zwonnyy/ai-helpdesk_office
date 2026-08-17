import argparse
from pathlib import Path

import joblib
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# ============================================================
# 경로
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

VALIDATION_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "validation.csv"
)

MODEL_ROOT = (
    BASE_DIR
    / "models"
)

REPORT_DIR = (
    BASE_DIR
    / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MAX_LENGTH = 256
BATCH_SIZE = 16


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Baseline VS Transformer"
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
    )

    return parser.parse_args()


# ============================================================
# 평가
# ============================================================

def evaluate(
    name,
    y_true,
    y_pred,
):

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
    )

    report = classification_report(
        y_true,
        y_pred,
        digits=4,
        zero_division=0,
    )

    print("\n")
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Accuracy    : "
        f"{accuracy:.4f}"
    )

    print(
        f"Macro F1    : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1 : "
        f"{weighted_f1:.4f}"
    )

    print("\n")
    print(
        "Classification Report"
    )
    print("-" * 70)

    print(report)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report": report,
    }


# ============================================================
# Transformer Prediction
# ============================================================

def transformer_predict(
    texts,
    transformer_path,
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nTransformer device: "
        f"{device}"
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            transformer_path
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            transformer_path
        )
    )

    model.to(
        device
    )

    model.eval()

    predictions = []

    texts = (
        texts
        .fillna("")
        .astype(str)
        .tolist()
    )

    total = len(texts)

    for start in range(
        0,
        total,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            total,
        )

        batch = texts[
            start:end
        ]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(
                device
            )
            for key, value
            in inputs.items()
        }

        with torch.inference_mode():

            outputs = model(
                **inputs
            )

        prediction_ids = (
            torch.argmax(
                outputs.logits,
                dim=-1,
            )
        )

        for class_id in (
            prediction_ids
            .cpu()
            .tolist()
        ):

            label = (
                model.config.id2label[
                    class_id
                ]
            )

            predictions.append(
                label
            )

        if (
            start % (BATCH_SIZE * 20)
            == 0
        ):

            print(
                f"Transformer 예측: "
                f"{end:,} / "
                f"{total:,}"
            )

    return predictions


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    target = args.target

    baseline_path = (
        MODEL_ROOT
        / f"{target}_baseline.joblib"
    )

    transformer_path = (
        MODEL_ROOT
        / f"{target}_transformer"
    )

    if not baseline_path.exists():

        raise FileNotFoundError(
            "Baseline 모델이 없습니다.\n"
            f"{baseline_path}"
        )

    if not transformer_path.exists():

        raise FileNotFoundError(
            "Transformer 모델이 없습니다.\n"
            f"{transformer_path}"
        )

    print("=" * 70)
    print(
        f"{target.upper()} MODEL COMPARISON"
    )
    print("=" * 70)

    df = pd.read_csv(
        VALIDATION_PATH
    )

    if target not in df.columns:

        raise ValueError(
            f"Validation 데이터에 "
            f"{target} 컬럼이 없습니다."
        )

    df = df.dropna(
        subset=[
            "text",
            target,
        ]
    ).copy()

    df["text"] = (
        df["text"]
        .astype(str)
    )

    y_true = (
        df[target]
    )

    # ========================================================
    # Baseline
    # ========================================================

    print("\nBaseline 모델 로드...")

    baseline = joblib.load(
        baseline_path
    )

    baseline_predictions = (
        baseline.predict(
            df["text"]
        )
    )

    baseline_result = evaluate(
        "BASELINE",
        y_true,
        baseline_predictions,
    )

    # ========================================================
    # Transformer
    # ========================================================

    print(
        "\nTransformer 모델 로드..."
    )

    transformer_predictions = (
        transformer_predict(
            df["text"],
            transformer_path,
        )
    )

    transformer_result = evaluate(
        "TRANSFORMER",
        y_true,
        transformer_predictions,
    )

    # ========================================================
    # 최종 비교
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(
        "Accuracy    : "
        f"{baseline_result['accuracy']:.4f}"
        " -> "
        f"{transformer_result['accuracy']:.4f}"
    )

    print(
        "Macro F1    : "
        f"{baseline_result['macro_f1']:.4f}"
        " -> "
        f"{transformer_result['macro_f1']:.4f}"
    )

    print(
        "Weighted F1 : "
        f"{baseline_result['weighted_f1']:.4f}"
        " -> "
        f"{transformer_result['weighted_f1']:.4f}"
    )

    # ========================================================
    # 오답 저장
    # ========================================================

    result_df = df[
        [
            "text",
            target,
        ]
    ].copy()

    result_df[
        "baseline_prediction"
    ] = baseline_predictions

    result_df[
        "transformer_prediction"
    ] = transformer_predictions

    result_df[
        "baseline_correct"
    ] = (
        result_df[target]
        == result_df[
            "baseline_prediction"
        ]
    )

    result_df[
        "transformer_correct"
    ] = (
        result_df[target]
        == result_df[
            "transformer_prediction"
        ]
    )

    prediction_path = (
        REPORT_DIR
        / (
            f"{target}_"
            f"model_predictions.csv"
        )
    )

    result_df.to_csv(
        prediction_path,
        index=False,
    )

    # Transformer 오답만 별도 저장

    transformer_errors = (
        result_df[
            ~result_df[
                "transformer_correct"
            ]
        ]
    )

    error_path = (
        REPORT_DIR
        / (
            f"{target}_"
            f"transformer_errors.csv"
        )
    )

    transformer_errors.to_csv(
        error_path,
        index=False,
    )

    # ========================================================
    # 비교 리포트 저장
    # ========================================================

    report_path = (
        REPORT_DIR
        / (
            f"{target}_"
            f"model_comparison.txt"
        )
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            f"TARGET: {target}\n\n"
        )

        file.write(
            "BASELINE\n"
        )

        file.write(
            "=" * 70
        )

        file.write("\n")

        file.write(
            baseline_result[
                "report"
            ]
        )

        file.write(
            "\n\nTRANSFORMER\n"
        )

        file.write(
            "=" * 70
        )

        file.write("\n")

        file.write(
            transformer_result[
                "report"
            ]
        )

        file.write(
            "\n\nFINAL COMPARISON\n"
        )

        file.write(
            "=" * 70
        )

        file.write("\n")

        file.write(
            "Accuracy: "
            f"{baseline_result['accuracy']:.4f}"
            " -> "
            f"{transformer_result['accuracy']:.4f}"
            "\n"
        )

        file.write(
            "Macro F1: "
            f"{baseline_result['macro_f1']:.4f}"
            " -> "
            f"{transformer_result['macro_f1']:.4f}"
            "\n"
        )

        file.write(
            "Weighted F1: "
            f"{baseline_result['weighted_f1']:.4f}"
            " -> "
            f"{transformer_result['weighted_f1']:.4f}"
            "\n"
        )

    print(
        f"\n비교 리포트: "
        f"{report_path}"
    )

    print(
        f"전체 예측 결과: "
        f"{prediction_path}"
    )

    print(
        f"Transformer 오답: "
        f"{error_path}"
    )


if __name__ == "__main__":
    main()