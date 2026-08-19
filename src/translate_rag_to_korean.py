import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data' / 'processed' / 'rag_train.csv'
OUTPUT_DIR = ROOT / 'data' / 'localized' / 'ko'
PARTIAL = OUTPUT_DIR / 'rag_train_ko.partial.csv'
FINAL = OUTPUT_DIR / 'rag_train_ko.csv'
MODEL_NAME = 'facebook/nllb-200-distilled-600M'
TARGET_CODE = 'kor_Hang'
TEXT_COLUMNS = ('subject', 'body', 'answer')
OUTPUT_COLUMNS = (
    'kb_index',
    'source_language',
    'subject_ko',
    'body_ko',
    'answer_ko',
    'translation_status',
    'error',
)
LANGUAGE_CODES = {
    'en': 'eng_Latn',
    'en-us': 'eng_Latn',
    'en-gb': 'eng_Latn',
    'english': 'eng_Latn',
    'de': 'deu_Latn',
    'de-de': 'deu_Latn',
    'german': 'deu_Latn',
}


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ''
    return str(value).strip()


def source_code(value: object) -> str | None:
    key = clean_text(value).lower().replace('_', '-')
    return LANGUAGE_CODES.get(key)


def token_count(tokenizer, text: str) -> int:
    return len(
        tokenizer(
            text,
            add_special_tokens=False,
        ).input_ids
    )


def chunk_text(
    text: str,
    tokenizer,
    max_tokens: int = 400,
) -> list[str]:
    if not text:
        return []
    if token_count(tokenizer, text) <= max_tokens:
        return [text]

    normalized = text.replace(chr(13), ' ').replace(chr(10), ' ')
    sentences = [
        sentence.strip()
        for sentence in re.split(r'(?<=[.!?]) +', normalized)
        if sentence.strip()
    ]
    chunks: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        candidate = ' '.join([*current, sentence])
        if current and token_count(tokenizer, candidate) > max_tokens:
            chunks.append(' '.join(current))
            current = []

        if token_count(tokenizer, sentence) <= max_tokens:
            current.append(sentence)
            continue

        words: list[str] = []
        for word in sentence.split():
            candidate = ' '.join([*words, word])
            if words and token_count(tokenizer, candidate) > max_tokens:
                chunks.append(' '.join(words))
                words = [word]
            else:
                words.append(word)
        if words:
            chunks.append(' '.join(words))

    if current:
        chunks.append(' '.join(current))
    return chunks or [text]


class NllbTranslator:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        print(f'Translation model: {MODEL_NAME}')
        print(f'Device: {self.device}')
        if self.device.type == 'cuda':
            print(f'GPU: {torch.cuda.get_device_name(0)}')

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            src_lang='eng_Latn',
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        self.target_id = self.tokenizer.convert_tokens_to_ids(TARGET_CODE)

    def translate_many(
        self,
        texts: list[str],
        language: str,
    ) -> list[str]:
        if not texts:
            return []
        self.tokenizer.src_lang = language
        chunks: list[str] = []
        owners: list[int] = []
        for owner, text in enumerate(texts):
            parts = chunk_text(text, self.tokenizer)
            chunks.extend(parts)
            owners.extend([owner] * len(parts))

        translated: list[str] = []
        for start in range(0, len(chunks), self.batch_size):
            encoded = self.tokenizer(
                chunks[start:start + self.batch_size],
                return_tensors='pt',
                padding=True,
                truncation=False,
            ).to(self.device)
            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    forced_bos_token_id=self.target_id,
                    max_length=512,
                    num_beams=1,
                )
            translated.extend(
                self.tokenizer.batch_decode(
                    generated,
                    skip_special_tokens=True,
                )
            )

        grouped: dict[int, list[str]] = defaultdict(list)
        for owner, value in zip(owners, translated, strict=True):
            grouped[owner].append(value.strip())
        return [
            ' '.join(grouped[index]).strip()
            for index in range(len(texts))
        ]


def atomic_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(temporary, index=False, encoding='utf-8-sig')
    temporary.replace(path)


def load_checkpoint() -> pd.DataFrame:
    path = PARTIAL if PARTIAL.exists() else FINAL
    if not path.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    frame = pd.read_csv(path, keep_default_na=False)
    missing = set(OUTPUT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing checkpoint columns: {sorted(missing)}')
    if frame['kb_index'].duplicated().any():
        raise ValueError('Checkpoint contains duplicate kb_index values.')
    return frame.loc[:, OUTPUT_COLUMNS]


def translate_rows(
    corpus: pd.DataFrame,
    indices: list[int],
    translator: NllbTranslator,
) -> list[dict]:
    records: dict[int, dict] = {}
    groups: dict[str, list[int]] = defaultdict(list)

    for index in indices:
        row = corpus.iloc[index]
        language = clean_text(row.get('language'))
        code = source_code(language)
        records[index] = {
            'kb_index': index,
            'source_language': language,
            'subject_ko': clean_text(row.get('subject')),
            'body_ko': clean_text(row.get('body')),
            'answer_ko': clean_text(row.get('answer')),
            'translation_status': 'skipped',
            'error': '',
        }
        if code:
            groups[code].append(index)
        else:
            records[index]['error'] = f'Unsupported language: {language}'

    for code, group in groups.items():
        try:
            for column in TEXT_COLUMNS:
                texts = [clean_text(corpus.at[index, column]) for index in group]
                positions = [pos for pos, text in enumerate(texts) if text]
                values = translator.translate_many(
                    [texts[pos] for pos in positions],
                    code,
                )
                for position, value in zip(positions, values, strict=True):
                    records[group[position]][f'{column}_ko'] = value
            for index in group:
                records[index]['translation_status'] = 'completed'
        except Exception as error:
            message = f'{type(error).__name__}: {error}'
            for index in group:
                records[index]['translation_status'] = 'failed'
                records[index]['error'] = message
    return [records[index] for index in indices]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create Korean presentation data for the RAG corpus.',
    )
    parser.add_argument('--limit', type=int)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint-every', type=int, default=50)
    return parser.parse_args()


def checkpoint_frame(records: dict[int, dict]) -> pd.DataFrame:
    return pd.DataFrame(
        records.values(),
        columns=OUTPUT_COLUMNS,
    ).sort_values('kb_index')


def main() -> None:
    args = arguments()
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be at least 1.')
    if args.batch_size < 1 or args.checkpoint_every < 1:
        raise ValueError('Batch and checkpoint sizes must be at least 1.')

    corpus = pd.read_csv(SOURCE)
    print(f'Source: {SOURCE}')
    print(f'Rows: {len(corpus):,}')
    print('Languages:')
    print(corpus['language'].value_counts(dropna=False).to_string())

    target = min(args.limit, len(corpus)) if args.limit else len(corpus)
    existing = (
        load_checkpoint()
        if args.resume
        else pd.DataFrame(columns=OUTPUT_COLUMNS)
    )
    if not args.resume and (PARTIAL.exists() or FINAL.exists()):
        raise FileExistsError(
            'Localization output exists. Use --resume to continue.'
        )

    records = {
        int(record['kb_index']): record
        for record in existing.to_dict('records')
    }
    complete = {
        index
        for index, record in records.items()
        if record['translation_status'] in ('completed', 'skipped')
    }
    pending = [index for index in range(target) if index not in complete]
    print(f'Target rows: {target:,}')
    print(f'Already complete: {target - len(pending):,}')
    print(f'Pending: {len(pending):,}')
    if not pending:
        return

    translator = NllbTranslator(args.batch_size)
    unsaved = 0
    try:
        for start in range(0, len(pending), args.batch_size):
            indices = pending[start:start + args.batch_size]
            for record in translate_rows(corpus, indices, translator):
                records[int(record['kb_index'])] = record
            unsaved += len(indices)
            done = min(start + len(indices), len(pending))
            print(f'Progress: {done:,}/{len(pending):,}')
            if unsaved >= args.checkpoint_every:
                atomic_write(checkpoint_frame(records), PARTIAL)
                unsaved = 0
    except (KeyboardInterrupt, Exception):
        atomic_write(checkpoint_frame(records), PARTIAL)
        print(f'Checkpoint saved: {PARTIAL}')
        raise

    frame = checkpoint_frame(records)
    atomic_write(frame, PARTIAL)
    if set(frame['kb_index'].astype(int)) >= set(range(len(corpus))):
        atomic_write(frame, FINAL)
        print(f'Completed localization: {FINAL}')
    else:
        print(f'Partial localization: {PARTIAL}')


if __name__ == '__main__':
    main()
