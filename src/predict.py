from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "type_baseline.joblib"
)


def load_model():
    print("모델을 불러옵니다.")

    model = joblib.load(MODEL_PATH)

    return model


def predict(model, text: str):
    prediction = model.predict([text])[0]

    probabilities = model.predict_proba([text])[0]

    classes = model.classes_

    results = sorted(
        zip(classes, probabilities),
        key=lambda x: x[1],
        reverse=True,
    )

    return prediction, results


def main():
    model = load_model()

    print()
    print("=" * 70)
    print("AI HELPDESK TYPE CLASSIFIER")
    print("=" * 70)

    while True:
        print()
        text = input(
            "문의 내용을 입력하세요 (종료: exit):\n> "
        )

        if text.lower() == "exit":
            print("종료합니다.")
            break

        if not text.strip():
            continue

        prediction, results = predict(
            model,
            text,
        )

        print()
        print(
            f"예측 결과: {prediction}"
        )

        print()
        print("확률")

        for label, probability in results:
            print(
                f"{label:10} "
                f"{probability * 100:6.2f}%"
            )


if __name__ == "__main__":
    main()