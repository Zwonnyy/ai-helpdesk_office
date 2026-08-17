from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_ROOT = BASE_DIR / "models"

TARGETS = [
    "type",
    "queue",
    "priority",
]

MAX_LENGTH = 256


def get_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def load_models(device):
    models = {}

    for target in TARGETS:
        model_path = (
            MODEL_ROOT
            / f"{target}_transformer"
        )

        print(
            f"{target.upper()} 모델 로드..."
        )

        tokenizer = (
            AutoTokenizer.from_pretrained(
                model_path
            )
        )

        model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                model_path
            )
        )

        model.to(device)
        model.eval()

        models[target] = {
            "tokenizer": tokenizer,
            "model": model,
        }

    return models


def predict_one(
    text,
    tokenizer,
    model,
    device,
):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    inputs = {
        key: value.to(device)
        for key, value
        in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1,
    )[0]

    best_id = torch.argmax(
        probabilities
    ).item()

    label = (
        model.config.id2label[
            best_id
        ]
    )

    confidence = (
        probabilities[
            best_id
        ].item()
    )

    return {
        "label": label,
        "confidence": confidence,
    }


def predict_all(
    text,
    models,
    device,
):
    results = {}

    for target in TARGETS:
        tokenizer = (
            models[target][
                "tokenizer"
            ]
        )

        model = (
            models[target][
                "model"
            ]
        )

        results[target] = predict_one(
            text,
            tokenizer,
            model,
            device,
        )

    return results


def main():
    device = get_device()

    print("=" * 70)
    print("AI HELPDESK")
    print("=" * 70)

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    models = load_models(
        device
    )

    print("\n모델 로드 완료.")

    while True:
        print()
        print("-" * 70)

        subject = input(
            "제목 (종료: exit): "
        )

        if subject.lower() == "exit":
            break

        body = input(
            "문의 내용: "
        )

        text = (
            subject.strip()
            + "\n"
            + body.strip()
        )

        results = predict_all(
            text,
            models,
            device,
        )

        print("\n")
        print("=" * 70)
        print("AI 분석 결과")
        print("=" * 70)

        for target in TARGETS:
            result = results[
                target
            ]

            print(
                f"{target.upper():10} : "
                f"{result['label']}"
                f" "
                f"({result['confidence'] * 100:.2f}%)"
            )


if __name__ == "__main__":
    main()