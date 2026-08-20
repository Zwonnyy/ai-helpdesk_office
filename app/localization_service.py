'''Deprecated CSV localization loader, no longer used by AnswerRetriever.'''

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
LOCALIZATION_DIR = BASE_DIR / 'data' / 'localized' / 'ko'
FINAL_PATH = LOCALIZATION_DIR / 'rag_train_ko.csv'
PARTIAL_PATH = LOCALIZATION_DIR / 'rag_train_ko.partial.csv'


def clean_optional(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


class KoreanLocalization:
    def __init__(self):
        self.rows: dict[int, dict] = {}
        self.path: Path | None = None

    def load(self, corpus_size: int) -> None:
        path = (
            FINAL_PATH
            if FINAL_PATH.exists()
            else PARTIAL_PATH
        )
        if not path.exists():
            print('Korean localization: unavailable (original text fallback)')
            return

        try:
            frame = pd.read_csv(path, keep_default_na=False)
            required = {
                'kb_index',
                'subject_ko',
                'body_ko',
                'answer_ko',
                'translation_status',
            }
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(
                    f'Missing columns: {sorted(missing)}'
                )
            if frame['kb_index'].duplicated().any():
                raise ValueError('Duplicate kb_index values')

            for row in frame.to_dict('records'):
                index = int(row['kb_index'])
                if 0 <= index < corpus_size:
                    self.rows[index] = row
            self.path = path
            print(
                f'Korean localization: {len(self.rows):,} rows '
                f'from {path.name}'
            )
        except Exception as error:
            self.rows = {}
            self.path = None
            print(
                'Korean localization load failed; '
                f'using original text. Error: {error}'
            )

    def resolve(
        self,
        index: int,
        subject: str,
        body: str,
        answer: str,
    ) -> dict:
        row = self.rows.get(index)
        if not row or row.get('translation_status') != 'completed':
            return {
                'subject_ko': subject,
                'body_ko': body,
                'answer_ko': answer,
                'translation_status': (
                    clean_optional(
                        row.get('translation_status')
                    )
                    if row
                    else None
                ),
            }

        return {
            'subject_ko': (
                clean_optional(row.get('subject_ko'))
                or subject
            ),
            'body_ko': (
                clean_optional(row.get('body_ko'))
                or body
            ),
            'answer_ko': (
                clean_optional(row.get('answer_ko'))
                or answer
            ),
            'translation_status': 'completed',
        }
