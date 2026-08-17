# AI Helpdesk

다국어 IT 헬프데스크 티켓을 자동 분석하고 과거 해결 사례를 검색해 답변 초안을 만드는 FastAPI 서비스입니다. 티켓의 유형(`type`), 담당 큐(`queue`), 우선순위(`priority`)를 분류하며, 담당자가 답변을 승인하거나 반려하는 Human-in-the-loop 워크플로를 제공합니다.

## 주요 기능

- 다국어 DistilBERT 기반 티켓 유형·담당 큐·우선순위 분류
- 파인튜닝된 Sentence Transformer 기반 유사 티켓 검색
- Ollama `qwen3:4b`를 이용한 고객 답변 초안 생성
- SQLite 기반 티켓 저장 및 상태 관리
- 분석 → 승인 요청 → 승인/반려 검토 워크플로
- 전처리, 학습, 평가, 오류 분석 스크립트와 실험 보고서

## 동작 구조

```text
새 티켓
  ├─ 분류 모델 ───────────────> type / queue / priority
  ├─ 임베딩 검색기 ───────────> 유사한 과거 티켓과 답변
  └─ Ollama(선택 기능) ───────> 고객 응답 초안
                                  │
                                  v
                    담당자 승인 또는 반려
```

API 시작 시 SQLite 테이블을 만들고 분류 모델 3개와 검색 모델을 메모리에 적재합니다. Ollama는 `/draft` 호출 시에만 필요합니다.

## 기술 스택

- Python 3.12+
- FastAPI, SQLAlchemy, SQLite
- PyTorch, Transformers, Sentence Transformers
- pandas, scikit-learn
- Ollama (`qwen3:4b`)
- uv

## 프로젝트 구조

```text
ai-helpdesk/
├── app/                    # FastAPI 애플리케이션과 서비스 계층
│   ├── main.py             # API 엔드포인트와 생명주기
│   ├── model_service.py    # 분류 모델 로딩 및 추론
│   ├── answer_retriever.py # 의미 기반 유사 티켓 검색
│   ├── llm_service.py      # Ollama 답변 초안 생성
│   ├── ticket_service.py   # 티켓 상태 전이와 비즈니스 로직
│   ├── database.py         # SQLite 연결 및 세션
│   ├── models.py           # SQLAlchemy 모델
│   └── schemas.py          # 요청·응답 스키마
├── data/
│   ├── raw/                # 원본 다국어 티켓 데이터
│   └── processed/          # 학습·검색용 전처리 데이터
├── models/                 # 모델과 임베딩(대용량, Git 제외)
├── reports/                # 평가 결과, 오류 사례, 시각화
├── src/                    # 전처리·학습·평가 스크립트
├── pyproject.toml          # 의존성 정의
└── uv.lock                 # 의존성 잠금 파일
```

## 설치

### 1. 저장소 복제 및 의존성 설치

[`uv`](https://docs.astral.sh/uv/)가 설치되어 있어야 합니다.

```bash
git clone https://github.com/Zwonnyy/ai-helpdesk_office.git
cd ai-helpdesk_office
uv sync
```

Windows에서는 잠금 파일 설정에 따라 CUDA 12.6용 PyTorch 인덱스를 사용합니다. CUDA가 없으면 CPU로 동작하지만 학습과 추론이 느릴 수 있습니다.

### 2. 모델 산출물 준비

`models/`는 GitHub 파일 크기 제한을 고려해 저장소에서 제외되어 있습니다. 서버 실행 전 다음 산출물이 필요합니다.

```text
models/
├── type_transformer/
├── queue_transformer/
├── priority_transformer/
├── helpdesk_embedding_model_v2/
└── rag_embeddings_finetuned_v2.npy
```

기존 산출물을 위 경로에 복사하거나 아래의 [모델 재학습](#모델-재학습) 절차로 생성합니다.

### 3. Ollama 준비(선택)

답변 초안 API를 사용하려면 Ollama를 실행하고 모델을 내려받습니다.

```bash
ollama pull qwen3:4b
```

Ollama 없이도 `/predict`, `/assist`, 티켓 분석 및 승인 API를 사용할 수 있습니다.

## 실행

```bash
uv run fastapi dev app/main.py
```

기본 주소는 `http://127.0.0.1:8000`입니다.

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- 상태 확인: `http://127.0.0.1:8000/health`

첫 실행 시 `data/helpdesk.db`가 자동 생성되며 이 런타임 DB는 Git에서 제외됩니다.

## API

| 메서드 | 경로 | 설명 |
|---|---|---|
| `GET` | `/` | 서비스 버전과 상태 확인 |
| `GET` | `/health` | 장치, 모델, 검색기, DB 상태 확인 |
| `POST` | `/predict` | 유형·큐·우선순위 예측 |
| `POST` | `/assist?top_k=3` | 예측 결과와 유사 티켓 반환 |
| `POST` | `/draft?top_k=3` | 예측·검색 기반 답변 초안 생성 |
| `POST` | `/tickets` | 티켓 생성 |
| `GET` | `/tickets` | 티켓 목록 조회(`offset`, `limit`) |
| `GET` | `/tickets/{ticket_id}` | 티켓 상세 조회 |
| `POST` | `/tickets/{ticket_id}/analyze` | 분류 및 유사 티켓 검색 |
| `POST` | `/tickets/{ticket_id}/submit-for-approval` | 초안을 승인 대기 상태로 제출 |
| `POST` | `/tickets/{ticket_id}/approve` | 최종 답변 승인 |
| `POST` | `/tickets/{ticket_id}/reject` | 사유와 함께 반려 |

### 호출 예시

```bash
curl -X POST "http://127.0.0.1:8000/assist?top_k=3" \
  -H "Content-Type: application/json" \
  -d '{"subject":"VPN connection unavailable","body":"All employees cannot connect to the VPN."}'
```

상태 전이는 다음과 같습니다.

```text
PENDING ──analyze──> ANALYZED ──submit-for-approval──> WAITING_APPROVAL
                           ^                              ├──approve──> APPROVED
                           └──────── analyze <── REJECTED <──reject───┘
```

반려된 티켓은 다시 분석하거나 수정한 초안을 재제출할 수 있습니다. 허용되지 않은 상태의 변경 요청은 `409 Conflict`를 반환합니다.

## 모델 재학습

GPU 사용을 권장합니다.

### 분류 모델

```bash
uv run python src/preprocess.py
uv run python src/train_transformer.py --target type
uv run python src/train_transformer.py --target queue
uv run python src/train_transformer.py --target priority
```

기본값은 3 epoch, batch size 8이며 `--epochs`, `--batch-size`로 조정합니다. 결과는 `models/{target}_transformer/`에 저장됩니다.

TF-IDF + Logistic Regression 기준 모델도 학습할 수 있습니다.

```bash
uv run python src/train_baseline.py --target type
uv run python src/train_baseline.py --target queue
uv run python src/train_baseline.py --target priority
```

### 의미 검색 모델

```bash
uv run python src/prepare_rag_data.py
uv run python src/build_embedding_retriever.py
uv run python src/prepare_semantic_pairs.py
uv run python src/train_helpdesk_embedding_v2.py
uv run python src/build_finetuned_embeddings_v2.py
```

RAG 데이터 분할, 초기 임베딩 생성, 의미 유사 문장 쌍 구성, Sentence Transformer 파인튜닝, 최종 검색 임베딩 생성을 차례로 수행합니다.

## 평가와 분석

```bash
uv run python src/compare_models.py --target type
uv run python src/evaluate_finetuned_retriever_v2.py
uv run python src/evaluate_answer_quality.py
uv run python src/analyze_retriever_comparison.py
```

결과는 주로 `reports/` 아래의 텍스트, CSV, PNG 파일로 저장됩니다.

## 데이터와 대용량 파일

- 원본 및 전처리 CSV는 `data/`에 있습니다.
- 가중치, 체크포인트, 벡터라이저, 임베딩은 `models/`에 생성됩니다.
- `models/` 전체는 `.gitignore`에 등록되어 커밋과 푸시에 포함되지 않습니다.
- 로컬 SQLite 데이터와 `.env` 파일도 Git에서 제외됩니다.

## 참고 사항

- 실제 서비스 진입점은 `app/main.py`이며 루트의 `main.py`는 PyCharm 예제입니다.
- 모델 또는 임베딩 파일이 없으면 애플리케이션 시작 단계에서 오류가 발생합니다.
- `/draft`에서 Ollama 실행에 실패하면 다른 기능에는 영향 없이 `503 Service Unavailable`을 반환합니다.
