from __future__ import annotations

import re
import threading
from collections import defaultdict
from typing import Any

import pandas as pd
import torch
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.models import HistoricalTicketTranslation


MODEL_NAME = 'facebook/nllb-200-distilled-600M'
TARGET_CODES = {'ko': 'kor_Hang'}
SOURCE_CODES = {
    'en': 'eng_Latn',
    'english': 'eng_Latn',
    'de': 'deu_Latn',
    'german': 'deu_Latn',
}
TEXT_FIELDS = ('subject', 'body', 'answer')


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ''
    return str(value).strip()


class TranslationService:
    def __init__(self, model_name: str = MODEL_NAME, max_chunk_tokens: int = 400):
        self.model_name = model_name
        self.max_chunk_tokens = max_chunk_tokens
        self.model = None
        self.tokenizer = None
        self.device: torch.device | None = None
        self._load_lock = threading.Lock()
        self._translation_lock = threading.Lock()

    def _load_model(self) -> None:
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:
                return
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            dtype = torch.float16 if device.type == 'cuda' else torch.float32
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, src_lang='eng_Latn',
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, dtype=dtype, low_cpu_mem_usage=True,
            )
            try:
                model = model.to(device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                device = torch.device('cpu')
                model = model.to(device=device, dtype=torch.float32)
            model.eval()
            self.tokenizer = tokenizer
            self.model = model
            self.device = device

    def _token_count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)

    def _chunks(self, text: str) -> list[str]:
        if not text:
            return []
        if self._token_count(text) <= self.max_chunk_tokens:
            return [text]
        sentences = [
            part.strip()
            for part in re.split(r'(?<=[.!?])\s+', text.replace('\r', ' ').replace('\n', ' '))
            if part.strip()
        ] or [text]
        chunks: list[str] = []
        current: list[str] = []
        for sentence in sentences:
            candidate = ' '.join([*current, sentence])
            if current and self._token_count(candidate) > self.max_chunk_tokens:
                chunks.append(' '.join(current))
                current = []
            if self._token_count(sentence) <= self.max_chunk_tokens:
                current.append(sentence)
                continue
            words: list[str] = []
            for word in sentence.split():
                if self._token_count(word) > self.max_chunk_tokens:
                    if words:
                        chunks.append(' '.join(words))
                        words = []
                    token_ids = self.tokenizer(
                        word, add_special_tokens=False,
                    ).input_ids
                    for start in range(0, len(token_ids), self.max_chunk_tokens):
                        chunks.append(self.tokenizer.decode(
                            token_ids[start:start + self.max_chunk_tokens],
                            skip_special_tokens=True,
                        ))
                    continue
                candidate = ' '.join([*words, word])
                if words and self._token_count(candidate) > self.max_chunk_tokens:
                    chunks.append(' '.join(words))
                    words = [word]
                else:
                    words.append(word)
            if words:
                chunks.append(' '.join(words))
        if current:
            chunks.append(' '.join(current))
        return chunks

    def _translate_texts(
        self, texts: list[str], source_code: str, target_code: str,
    ) -> list[str]:
        self._load_model()
        with self._translation_lock:
            self.tokenizer.src_lang = source_code
            chunks: list[str] = []
            owners: list[int] = []
            for owner, value in enumerate(texts):
                parts = self._chunks(value)
                chunks.extend(parts)
                owners.extend([owner] * len(parts))
            if not chunks:
                return ['' for _ in texts]
            target_id = self.tokenizer.convert_tokens_to_ids(target_code)
            translated: list[str] = []
            for chunk in chunks:
                encoded = self.tokenizer(
                    chunk, return_tensors='pt', truncation=False,
                ).to(self.device)
                with torch.inference_mode():
                    output = self.model.generate(
                        **encoded,
                        forced_bos_token_id=target_id,
                        max_new_tokens=512,
                        num_beams=1,
                    )
                translated.append(
                    self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
                )
            grouped: dict[int, list[str]] = defaultdict(list)
            for owner, value in zip(owners, translated, strict=True):
                grouped[owner].append(value)
            return [' '.join(grouped[index]).strip() for index in range(len(texts))]

    def get_or_translate(
        self, db: Session, similar_ticket: dict, target_language: str = 'ko',
    ) -> dict:
        kb_index = similar_ticket.get('kb_index')
        originals = {field: clean_text(similar_ticket.get(field)) for field in TEXT_FIELDS}
        base = {'kb_index': kb_index, 'target_language': target_language}
        if kb_index is None:
            return {
                **base, **originals, 'cached': False, 'translated': False,
                'error': '이전 분석 결과에 kb_index가 없어 번역할 수 없습니다.',
            }
        try:
            kb_index = int(kb_index)
            base['kb_index'] = kb_index
        except (TypeError, ValueError):
            return {
                **base, **originals, 'cached': False, 'translated': False,
                'error': '유효하지 않은 kb_index라 번역할 수 없습니다.',
            }
        target_code = TARGET_CODES.get(target_language.lower())
        if target_code is None:
            return {
                **base, **originals, 'cached': False, 'translated': False,
                'error': f'지원하지 않는 대상 언어입니다: {target_language}',
            }
        cached = db.scalar(
            select(HistoricalTicketTranslation).where(
                HistoricalTicketTranslation.kb_index == int(kb_index),
                HistoricalTicketTranslation.target_language == target_language,
            )
        )
        if cached is not None:
            return {
                **base,
                'subject': cached.subject_translated,
                'body': cached.body_translated,
                'answer': cached.answer_translated,
                'cached': True, 'translated': True, 'error': None,
            }
        source_language = clean_text(similar_ticket.get('language')).lower()
        source_code = SOURCE_CODES.get(source_language)
        if source_code is None:
            source_label = source_language or 'unknown'
            return {
                **base, **originals, 'cached': False, 'translated': False,
                'error': f'지원하지 않는 원문 언어입니다: {source_label}',
            }
        try:
            values = self._translate_texts(
                [originals[field] for field in TEXT_FIELDS], source_code, target_code,
            )
            db.add(HistoricalTicketTranslation(
                kb_index=int(kb_index),
                target_language=target_language,
                subject_translated=values[0],
                body_translated=values[1],
                answer_translated=values[2],
                source_language=source_language,
            ))
            try:
                db.commit()
            except IntegrityError:
                db.rollback()
                return self.get_or_translate(db, similar_ticket, target_language)
            return {
                **base, 'subject': values[0], 'body': values[1], 'answer': values[2],
                'cached': False, 'translated': True, 'error': None,
            }
        except Exception as error:
            db.rollback()
            if isinstance(error, torch.cuda.OutOfMemoryError):
                try:
                    self.model = self.model.to(device='cpu', dtype=torch.float32)
                    self.device = torch.device('cpu')
                except Exception:
                    self.model = None
                    self.tokenizer = None
                    self.device = None
                torch.cuda.empty_cache()
            print(f'Translation failed for kb_index={kb_index}: {type(error).__name__}: {error}')
            return {
                **base, **originals, 'cached': False, 'translated': False,
                'error': '번역에 실패했습니다. 원문을 반환합니다.',
            }
