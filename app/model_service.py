from pathlib import Path

import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

MODEL_ROOT = (
    BASE_DIR
    / "models"
)

TARGETS = [
    "type",
    "queue",
    "priority",
]

MAX_LENGTH = 256


class ModelService:

    def __init__(self):
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.models = {}

    def load_models(self):
        print("=" * 70)
        print("AI HELPDESK MODEL LOADING")
        print("=" * 70)

        print(
            f"Device: {self.device}"
        )

        if torch.cuda.is_available():
            print(
                "GPU:",
                torch.cuda.get_device_name(0),
            )

        for target in TARGETS:
            self._load_model(
                target
            )

        print(
            "\n모든 모델 로드 완료."
        )

    def _load_model(
        self,
        target: str,
    ):
        model_path = (
            MODEL_ROOT
            / f"{target}_transformer"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"{target} 모델이 없습니다.\n"
                f"{model_path}"
            )

        print(
            f"\n{target.upper()} "
            f"모델 로드..."
        )

        tokenizer = (
            AutoTokenizer
            .from_pretrained(
                model_path
            )
        )

        model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                model_path
            )
        )

        model.to(
            self.device
        )

        model.eval()

        self.models[target] = {
            "tokenizer": tokenizer,
            "model": model,
        }

        print(
            f"{target.upper()} "
            f"로드 완료."
        )

    def _predict_one(
        self,
        target: str,
        text: str,
    ):
        bundle = (
            self.models[target]
        )

        tokenizer = (
            bundle["tokenizer"]
        )

        model = (
            bundle["model"]
        )

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )

        inputs = {
            key: value.to(
                self.device
            )
            for key, value
            in inputs.items()
        }

        with torch.inference_mode():
            outputs = model(
                **inputs
            )

        probabilities = (
            torch.softmax(
                outputs.logits,
                dim=-1,
            )[0]
        )

        best_id = (
            torch.argmax(
                probabilities
            ).item()
        )

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

    def predict(
        self,
        subject: str,
        body: str,
    ):
        text = (
            subject.strip()
            + "\n"
            + body.strip()
        )

        results = {}

        for target in TARGETS:
            results[target] = (
                self._predict_one(
                    target,
                    text,
                )
            )

        return results