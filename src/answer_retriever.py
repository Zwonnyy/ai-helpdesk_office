from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import (
    linear_kernel,
)


BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

CORPUS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "answer_corpus.csv"
)

RETRIEVER_PATH = (
    BASE_DIR
    / "models"
    / "answer_retriever.joblib"
)


class AnswerRetriever:

    def __init__(self):

        self.vectorizer = None
        self.matrix = None
        self.corpus = None

        self.loaded = False

    # ========================================================
    # 모델 로드
    # ========================================================

    def load(self):

        print("\n")
        print("=" * 70)
        print("ANSWER RETRIEVER LOADING")
        print("=" * 70)

        if not CORPUS_PATH.exists():

            raise FileNotFoundError(
                "answer_corpus.csv가 없습니다.\n"
                f"{CORPUS_PATH}\n\n"
                "먼저 다음을 실행하세요:\n"
                "uv run python "
                "src/prepare_answers.py"
            )

        if not RETRIEVER_PATH.exists():

            raise FileNotFoundError(
                "answer_retriever.joblib이 "
                "없습니다.\n"
                f"{RETRIEVER_PATH}"
            )

        self.corpus = pd.read_csv(
            CORPUS_PATH
        )

        bundle = joblib.load(
            RETRIEVER_PATH
        )

        self.vectorizer = (
            bundle["vectorizer"]
        )

        self.matrix = (
            bundle["matrix"]
        )

        self.loaded = True

        print(
            f"Corpus: "
            f"{len(self.corpus):,}건"
        )

        print(
            f"Matrix: "
            f"{self.matrix.shape}"
        )

        print(
            "ANSWER RETRIEVER READY"
        )

    # ========================================================
    # 검색
    # ========================================================

    def search(
        self,
        subject: str,
        body: str,
        top_k: int = 3,
    ):

        if not self.loaded:
            raise RuntimeError(
                "AnswerRetriever가 "
                "로드되지 않았습니다."
            )

        text = (
            subject.strip()
            + "\n"
            + body.strip()
        )

        query_vector = (
            self.vectorizer.transform(
                [text]
            )
        )

        similarities = (
            linear_kernel(
                query_vector,
                self.matrix,
            )
            .flatten()
        )

        top_k = min(
            top_k,
            len(similarities),
        )

        # 높은 similarity 순으로 정렬
        top_indices = (
            np.argsort(
                similarities
            )[::-1][:top_k]
        )

        results = []

        for index in top_indices:

            row = self.corpus.iloc[
                index
            ]

            score = float(
                similarities[index]
            )

            results.append(
                {
                    "score": score,

                    "subject": (
                        str(
                            row["subject"]
                        )
                    ),

                    "body": (
                        str(
                            row["body"]
                        )
                    ),

                    "answer": (
                        str(
                            row["answer"]
                        )
                    ),

                    "type": (
                        str(
                            row["type"]
                        )
                    ),

                    "queue": (
                        str(
                            row["queue"]
                        )
                    ),

                    "priority": (
                        str(
                            row["priority"]
                        )
                    ),

                    "language": (
                        str(
                            row["language"]
                        )
                    ),
                }
            )

        return results