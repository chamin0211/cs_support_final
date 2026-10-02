# 로컬 LLM 은행 상담 보조 AI (cs_support_final)

고객 문의를 **주제 분류**해 전문 에이전트 3개(**잔액조회 · 대출문의 · 이자/연체**)로 연결하는 은행 상담 챗봇 MVP입니다.
모든 모델은 AI Hub 71926 은행 상담 데이터를 활용해 파인튜닝한 **로컬 양자화 모델(Qwen 계열, GGUF)** 이며, Ollama로 Windows(CUDA)·macOS(Metal) 양쪽에서 구동됩니다. 고객 정보는 모델에 도달하기 전에 마스킹되고, 금액 같은 실제 값은 화면에서만 치환됩니다.

> **내 역할 — 대출문의 에이전트 (`agent-loan`) 담당**
> 대출 mock API, 에이전트 로직(`handle()`), 출력 검증기, 학습 데이터 구성, `cs-loan` 모델 QLoRA 학습·GGUF 변환·평가(Golden Set·사전 등록 평가 기준)까지 에이전트 하나를 설계부터 평가까지 전 과정 담당했습니다. 상세는 [아래 섹션](#내-담당-대출문의-에이전트-agent-loan)을 참고하세요.

<!-- TODO: 데모 스크린샷/GIF 추가
## 데모
![챗봇 대출 문의 화면](docs/images/demo-loan.png)
-->

## 주요 특징

- **주제 분류 → 에이전트 연결**: 9개 상담 주제 중 지원하는 3개는 전문 에이전트가 답하고, 나머지는 상담원 연결 안내. 두 주제에 걸친 문의는 되묻기
- **개인정보 마스킹**: 주민등록번호·카드번호·계좌번호·전화번호·주소·금액 원본이 프롬프트에 들어가지 않음 (모델 입력 로그로 검증 가능)
- **슬롯 치환 방식**: 모델은 `{{principal_remaining}}` 같은 자리표시자만 생성하고, 실제 값은 응답의 `slots`로 프론트에 전달해 화면에서만 치환
- **지어내지 않는 답변**: 금리 수치·실제 상품명·없는 서류 요건을 만들지 않고, 모르면 상담원 확인을 안내
- **단일 모델 호출 지점**: 모든 모델 호출은 `backend/app/llm`의 `generate()`만 거쳐 입력 로그를 한 곳에서 기록

## 아키텍처

```
브라우저 (Next.js)
   │  POST /api/chat  (프론트가 호출하는 유일한 API)
   ▼
FastAPI 게이트웨이 ── 마스킹(masking) ── 라우터(cs-router, 주제 분류)
                                          │
                 ┌────────────────────────┼────────────────────────┐
                 ▼                        ▼                        ▼
          잔액조회 에이전트          대출문의 에이전트          이자/연체 에이전트
          (cs-balance)              (cs-loan)                 (cs-interest)
                 │                        │                        │
              mock 계좌 API            mock 대출 API            mock 이자 API
                 └──────────── llm.generate() → Ollama ────────────┘
```

## 기술 스택

| 영역 | 사용 기술 |
|------|-----------|
| Frontend | Next.js (App Router), TypeScript strict, Tailwind CSS, Vitest |
| Backend | Python 3.11, FastAPI, pytest |
| 모델 추론 | Ollama (GGUF Q4_K_M) |
| 학습 | Hugging Face Transformers, PEFT(QLoRA), TRL, llama.cpp(GGUF 변환) |
| 베이스 모델 | Qwen/Qwen3-4B-Instruct-2507 (대출문의) |

## 디렉토리 구조

```
backend/
├── app/
│   ├── gateway/     # /api/chat, 세션, 에이전트 디스패치
│   ├── masking/     # 개인정보 마스킹
│   ├── router/      # 주제 분류
│   ├── llm/         # generate() — 모델 호출 단일 진입점
│   └── agents/      # balance · loan · interest (+ base.py 공통 계약)
├── models/          # 모델별 Modelfile
├── training/        # 전처리·학습·평가 스크립트
└── tests/           # gateway · masking · llm · router · balance · loan · interest
frontend/            # 랜딩 + 챗봇 UI
docs/                # 공통 문서 + 영역별 PRD·ARCHITECTURE·ADR
phases/              # 단계별 구현 기록
presentation/        # 발표 자료
```

## 내 담당: 대출문의 에이전트 (`agent-loan`)

대출 만기·연장·조회 문의에 mock 대출 API를 조회해 존댓말 상담 답변을 돌려주는 에이전트와 `cs-loan` 모델을 만들었습니다.

- 코드: [`backend/app/agents/loan/`](backend/app/agents/loan) · 테스트: [`backend/tests/loan/`](backend/tests/loan) · 학습/평가: [`backend/training/loan/`](backend/training/loan) · 문서: [`docs/agent-loan/`](docs/agent-loan)

### 처리 흐름

1. `get_loans(customer_id)`로 mock 대출 조회
2. 대출이 없으면 **모델을 호출하지 않고** 고정 문장 반환 (없는 대출을 지어낼 여지 제거)
3. 슬롯 생성: `{{loan_label}}`(대출 종류), `{{principal_remaining}}`(남은 원금), `{{extendable_status}}`(연장 가능 여부 문구) — 값은 항상 mock 조회 결과에서 코드가 만듦
4. 프롬프트에는 금액을 넣지 않고(만기일·연장 여부 등 비금액 정보만) `llm.generate("cs-loan", ...)` 호출
5. **출력 검증(허용 목록 방식)**: 프롬프트에 준 만기일·마스킹 토큰 외의 숫자, 허용 밖 슬롯, 구체적 서류명, 없는 채널(앱·콜센터 등), 못 지키는 약속, 연장 불가인데 "가능"이라는 모순이 있으면 안전한 기본 문장으로 대체

### 문제 해결 과정: "지어내는 모델"을 어떻게 고쳤나

은행 상담에서 지어낸 금리·서류 요건은 치명적이라, 이 에이전트의 핵심 과제는 **사실을 지어내지 않게 만드는 것**이었습니다.

| 단계 | 발견 / 판단 | 조치 |
|------|-------------|------|
| v1 | AI Hub 상담사 답변으로 학습. 학습 데이터 30건을 점검하니 안전한 답변이 없었음 (우리 서비스에 없는 앱·지점·ARS 절차, 못 지키는 약속, 지어낸 서류·금리). v1 모델도 서류·금리·수수료 질문 70건 중 **16건**을 지어냄 | 데이터 자체가 원인이라 판단 |
| v2 | 답변을 학습 데이터에 쓰면 안 된다고 결론 | **AI Hub는 질문 공급원으로만 쓰고, 정답은 직접 작성**. 서류·금리 질문엔 "상담원 확인" 안내 문장으로 대응. Golden Set 57문항 통과 |
| v3 | v2 평가에서 복합 질문 시 원금 누락, 수수료 질문에 금리로 답변 등 약점 발견 | 수동 시드 164 → 350건 확장, 복합·전반 조회 문형 추가. 새 문항 통과 10 → 18/20 |
| v4 | v3에 지어낸 사실, "직접 처리해 드리겠다" 같은 처리 약속이 잔존 | 시드 451건으로 보강. 서비스 설정(temperature 0.3) 5회 반복(200샘플) 평가에서 **지어낸 사실 4건 → 0건**, 뒤집힌 거절 문형 24~41건 → 1건 |

- 모델 한계는 코드로 보완: 모델이 틀려도 화면에 틀린 사실이 나오지 않도록, 연장 가능 여부·원금 등은 모델이 아니라 **슬롯 값(mock 조회값)** 으로 표시하고, 출력 검증기가 이중 안전장치 역할을 합니다.
- 평가 방법: 학습에 쓰지 않은 Golden Set을 따로 두고, 판정 기준을 평가 **전에** 문서로 미리 고정(`EVAL_CRITERIA_v3/v4.md`)한 뒤 결과를 판정해 결과에 맞춘 기준 조정을 피했습니다.
- 학습 파이프라인: `Qwen3-4B-Instruct-2507` → QLoRA → bf16 베이스에 병합 → GGUF Q4_K_M → Ollama `cs-loan` 등록 ([`export.md`](backend/training/loan/export.md)). 학습·추론 모두 로컬 환경에서 수행합니다.
- **알려진 한계**: 낯선 날짜 표현("마지막 상환일")에서 과다 거절, "조건 + 해당 여부" 복합 질문에서 연장 슬롯 누락, 문형 정형화. v4 확정 근거와 트레이드오프는 [`docs/agent-loan/ADR.md`](docs/agent-loan/ADR.md)(LN-007~009)에 기록했습니다.

### mock 데이터

| customer_id | 대출 | 만기일 | 연장 |
|-------------|------|--------|------|
| `C001` | 없음 | - | - |
| `C002` | 신용대출 (`L001`) | 2027-03-31 | 가능 |
| `C003` | 주택담보대출 (`L002`) | 2035-06-30 | 불가 |

## 실행 방법

> 모델이 없어도 서버는 뜹니다. 라우터는 키워드 규칙으로 폴백하고, 에이전트 모델 호출 실패 시 "상담원 연결" 안내를 반환합니다.

```bash
# 1. 백엔드 환경 (최초 1회)
uv venv backend/.venv --python 3.11
uv pip install --python backend/.venv/bin/python -r backend/requirements.txt

# 2. 모델 등록 (.gguf 파일은 별도 전달받아 Modelfile 경로에 둠)
ollama serve
ollama create cs-router   -f backend/models/router/Modelfile
ollama create cs-balance  -f backend/models/balance/Modelfile
ollama create cs-loan     -f backend/models/loan/Modelfile
ollama create cs-interest -f backend/models/interest/Modelfile

# 3. API 서버 → http://localhost:8000/docs
cd backend && .venv/bin/uvicorn app.main:app --reload --port 8000

# 4. 챗봇 UI → http://localhost:3000   (Node 24+)
cd frontend && npm ci && npm run dev
```

동작 확인:

```bash
curl -s -X POST localhost:8000/api/chat -H 'Content-Type: application/json' \
  -d '{"session_id":"s1","customer_id":"C002","message":"제 대출 만기가 언제예요?"}'
```

## 테스트

```bash
cd backend && pytest                 # 전체
cd backend && pytest tests/loan      # 대출문의 에이전트 (모델 호출은 목으로 대체)
cd frontend && npm run test          # Vitest
cd frontend && npm run lint
```

## 팀 구성

| 영역 | 브랜치 |
|------|--------|
| 프론트엔드 + 잔액조회 에이전트 | `feat-frontend`, `feat-agent-balance` |
| **대출문의 에이전트** | `feat-agent-loan` |
| 이자/연체 에이전트 | `feat-agent-interest` |
| 라우터 + 게이트웨이 | `feat-router` |

영역 간 계약(`/api/chat` 스키마, `agents/base.py`, 주제 코드, 마스킹 토큰)은 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)에 정의되어 있습니다. 개발은 TDD와 conventional commits를 따랐습니다.

## 데이터·모델 관련 안내

- AI Hub 원본·가공 데이터와 모델 가중치(`.gguf`, `.safetensors`)는 데이터셋 이용조건(제3자 제공 금지)에 따라 **저장소에 포함되어 있지 않습니다.**
- 저장소의 학습용 JSONL(수동 시드·Golden Set)은 팀이 직접 작성한 문장입니다.

## Contact

- GitHub: [@chamin0211](https://github.com/chamin0211)
