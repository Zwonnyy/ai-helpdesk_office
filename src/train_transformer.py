import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets import Dataset

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# ============================================================
# 기본 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

MODEL_ROOT = (
    BASE_DIR
    / "models"
)

REPORT_DIR = (
    BASE_DIR
    / "reports"
)

TRAIN_PATH = (
    PROCESSED_DIR
    / "train.csv"
)

VALIDATION_PATH = (
    PROCESSED_DIR
    / "validation.csv"
)

MODEL_NAME = (
    "distilbert/"
    "distilbert-base-multilingual-cased"
)

MAX_LENGTH = 256

MODEL_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Multilingual DistilBERT "
            "Fine-tuning"
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

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
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

    for column in [
        "text",
        target,
    ]:

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

    train_df = (
        train_df.dropna(
            subset=[
                "text",
                target,
            ]
        )
        .copy()
    )

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

    labels = sorted(
        train_df[target]
        .unique()
        .tolist()
    )

    label2id = {
        label: index
        for index, label
        in enumerate(labels)
    }

    id2label = {
        index: label
        for label, index
        in label2id.items()
    }

    train_df["label"] = (
        train_df[target]
        .map(label2id)
    )

    validation_df["label"] = (
        validation_df[target]
        .map(label2id)
    )

    if (
        validation_df["label"]
        .isna()
        .any()
    ):
        raise ValueError(
            "Validation 데이터에 "
            "Train에 없는 클래스가 있습니다."
        )

    train_df["label"] = (
        train_df["label"]
        .astype(int)
    )

    validation_df["label"] = (
        validation_df["label"]
        .astype(int)
    )

    print(
        f"Target       : {target}"
    )

    print(
        f"Train        : "
        f"{len(train_df):,}"
    )

    print(
        f"Validation   : "
        f"{len(validation_df):,}"
    )

    print(
        f"클래스 수    : "
        f"{len(labels)}"
    )

    print("\nLabels")

    for index, label in enumerate(
        labels
    ):
        print(
            f"{index:2} → {label}"
        )

    return (
        train_df,
        validation_df,
        labels,
        label2id,
        id2label,
    )


# ============================================================
# Hugging Face Dataset
# ============================================================

def create_datasets(
    train_df,
    validation_df,
):

    print("\n")
    print("=" * 70)
    print("2. Hugging Face Dataset 생성")
    print("=" * 70)

    train_dataset = (
        Dataset.from_pandas(
            train_df[
                [
                    "text",
                    "label",
                ]
            ],
            preserve_index=False,
        )
    )

    validation_dataset = (
        Dataset.from_pandas(
            validation_df[
                [
                    "text",
                    "label",
                ]
            ],
            preserve_index=False,
        )
    )

    return (
        train_dataset,
        validation_dataset,
    )


# ============================================================
# Tokenizer
# ============================================================

def load_tokenizer():

    print("\n")
    print("=" * 70)
    print("3. Tokenizer")
    print("=" * 70)

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    return tokenizer


def tokenize_dataset(
    dataset,
    tokenizer,
):

    def tokenize(
        batch,
    ):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=[
            "text"
        ],
    )

    return tokenized


# ============================================================
# Model
# ============================================================

def build_model(
    labels,
    label2id,
    id2label,
):

    print("\n")
    print("=" * 70)
    print("4. Transformer 모델 생성")
    print("=" * 70)

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )
    )

    return model


# ============================================================
# Metrics
# ============================================================

def compute_metrics(
    eval_pred,
):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
    )

    weighted_f1 = f1_score(
        labels,
        predictions,
        average="weighted",
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    target = args.target

    epochs = args.epochs

    batch_size = args.batch_size

    print("=" * 70)
    print("GPU CHECK")
    print("=" * 70)

    cuda_available = (
        torch.cuda.is_available()
    )

    print(
        f"CUDA 사용 가능: "
        f"{cuda_available}"
    )

    if cuda_available:

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

        print(
            "GPU Memory:",
            round(
                torch.cuda.get_device_properties(
                    0
                ).total_memory
                / 1024**3,
                2,
            ),
            "GB",
        )

    else:

        print(
            "WARNING: CPU로 학습합니다."
        )

    (
        train_df,
        validation_df,
        labels,
        label2id,
        id2label,
    ) = load_data(
        target
    )

    (
        train_dataset,
        validation_dataset,
    ) = create_datasets(
        train_df,
        validation_df,
    )

    tokenizer = load_tokenizer()

    print("\nTokenizing Train...")

    train_dataset = tokenize_dataset(
        train_dataset,
        tokenizer,
    )

    print(
        "\nTokenizing Validation..."
    )

    validation_dataset = (
        tokenize_dataset(
            validation_dataset,
            tokenizer,
        )
    )

    model = build_model(
        labels,
        label2id,
        id2label,
    )

    data_collator = (
        DataCollatorWithPadding(
            tokenizer=tokenizer
        )
    )

    model_dir = (
        MODEL_ROOT
        / f"{target}_transformer"
    )

    checkpoint_dir = (
        model_dir
        / "checkpoints"
    )

    print("\n")
    print("=" * 70)
    print("5. Training 설정")
    print("=" * 70)

    print(
        f"Epoch       : {epochs}"
    )

    print(
        f"Batch Size  : {batch_size}"
    )

    print(
        f"Max Length  : {MAX_LENGTH}"
    )

    print(
        f"FP16        : "
        f"{cuda_available}"
    )

    training_args = (
        TrainingArguments(
            output_dir=str(
                checkpoint_dir
            ),

            learning_rate=2e-5,

            per_device_train_batch_size=(
                batch_size
            ),

            per_device_eval_batch_size=16,

            num_train_epochs=epochs,

            weight_decay=0.01,

            eval_strategy="epoch",

            save_strategy="epoch",

            load_best_model_at_end=True,

            metric_for_best_model=(
                "macro_f1"
            ),

            greater_is_better=True,

            save_total_limit=2,

            logging_steps=100,

            report_to="none",

            fp16=cuda_available,

            seed=42,
        )
    )

    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=(
            train_dataset
        ),

        eval_dataset=(
            validation_dataset
        ),

        processing_class=(
            tokenizer
        ),

        data_collator=(
            data_collator
        ),

        compute_metrics=(
            compute_metrics
        ),
    )

    print("\n")
    print("=" * 70)
    print("6. FINE-TUNING START")
    print("=" * 70)

    trainer.train()

    print("\n")
    print("=" * 70)
    print("7. Validation 평가")
    print("=" * 70)

    metrics = trainer.evaluate()

    for key, value in (
        metrics.items()
    ):

        if isinstance(
            value,
            float,
        ):

            print(
                f"{key}: "
                f"{value:.4f}"
            )

        else:

            print(
                f"{key}: "
                f"{value}"
            )

    prediction_output = (
        trainer.predict(
            validation_dataset
        )
    )

    predictions = np.argmax(
        prediction_output.predictions,
        axis=-1,
    )

    true_labels = (
        prediction_output.label_ids
    )

    report = classification_report(
        true_labels,
        predictions,
        labels=list(
            range(
                len(labels)
            )
        ),
        target_names=labels,
        digits=4,
        zero_division=0,
    )

    print("\n")
    print(
        "Classification Report"
    )
    print("-" * 70)

    print(report)

    report_path = (
        REPORT_DIR
        / (
            f"{target}_"
            f"transformer_"
            f"validation_report.txt"
        )
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            f"Target: {target}\n\n"
        )

        for key, value in (
            metrics.items()
        ):
            file.write(
                f"{key}: {value}\n"
            )

        file.write(
            "\n"
        )

        file.write(
            report
        )

    print(
        f"\n리포트 저장: "
        f"{report_path}"
    )

    print("\n")
    print("=" * 70)
    print("8. 모델 저장")
    print("=" * 70)

    trainer.save_model(
        str(
            model_dir
        )
    )

    tokenizer.save_pretrained(
        str(
            model_dir
        )
    )

    print(
        f"모델 저장 완료: "
        f"{model_dir}"
    )

    print("\n")
    print("=" * 70)
    print(
        f"{target.upper()} "
        f"TRANSFORMER TRAINING COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()