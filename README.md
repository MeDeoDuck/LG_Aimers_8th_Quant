# LG Aimers 8th : LLM Compression — EXAONE-4.0-1.2B를 4비트 양자화해 추론 단가를 낮추는 경량화 실험

서빙·온디바이스 환경에서 LLM 추론 비용을 줄이기 위해, LG의 sLLM(EXAONE-4.0-1.2B)을 GPTQ W4A16 양자화와 구조적 pruning, knowledge distillation으로 압축한 LG Aimers 8기 모델 경량화 해커톤 실험 저장소입니다.

![Program](https://img.shields.io/badge/LG%20Aimers-8th-red)
![Track](https://img.shields.io/badge/track-LLM%20Compression-blue)
![Method](https://img.shields.io/badge/method-GPTQ%20W4A16-green)
![Base](https://img.shields.io/badge/base-EXAONE--4.0--1.2B-orange)
![Type](https://img.shields.io/badge/repo-Fork-lightgrey)

> ⚠️ **이 저장소는 Fork입니다.** 원본은 `panhong99/LG_Aimers`이며, 본 README는 **본인이 직접 작성·추가한 실험 스크립트**(`seed_equal.py`, `scale_pruning_distill.py`, `nearly_zero.py`, `ignore_dummy_GPTQ.py`, `estimate_score.py`)만 서술합니다. `default.py` / `default.ipynb`는 대회에서 제공한 baseline 코드로, 본인 작성분이 아님을 명시합니다.

---

## 🎯 소개

**LG Aimers 8기 모델 경량화 온라인 해커톤(Dacon)** 트랙에서 진행한 LLM 양자화 실험입니다. LG AI Research의 `LGAI-EXAONE/EXAONE-4.0-1.2B`(30 레이어) 모델을 대상으로, 성능 손실을 억제하면서 추론 속도를 끌어올리는 경량화 방법을 코드로 검증했습니다.

핵심 접근은 **GPTQ W4A16(4비트 가중치 양자화)** 이며, 여기에 채널 단위 pruning, knowledge distillation, 특정 레이어 무력화·양자화 제외 등 여러 변형을 스크립트로 나누어 실험했습니다. LG Aimers 8기 프로그램에 참여해 수료했습니다.

- 🧪 **양자화 baseline에서 출발한 다변형 실험**: 시드 고정, pruning + 증류, 레이어 무력화, 부분 양자화 제외
- 📉 **목표는 추론 단가 절감**: 4비트 압축으로 모델 크기·토큰당 생성 시간을 낮추는 방향
- 🧮 **자체 점수 추정 하네스**: 대회 채점식(성능 + 속도)을 로컬에서 근사하는 `estimate_score.py` 작성
- 🔌 **서빙 호환 보정**: safetensors 가중치 키 정규화로 vLLM 로딩 호환성 확보

---

## 🧩 목표 (추론 비용 절감)

sLLM이라도 서빙 시 토큰당 연산·메모리 비용이 누적되므로, 정확도 저하를 최소화하면서 다음을 낮추는 것이 목표입니다.

- **모델 메모리**: FP/BF16 가중치를 4비트(W4A16)로 압축해 로딩·상주 메모리 축소
- **토큰당 생성 시간**: 저비트 커널(Marlin 경로)과 불필요 연산 제거로 처리량 확보
- **정확도 보존**: 입출력단(`embed_tokens`, `lm_head`)과 민감 레이어를 선별 제외해 압축에 따른 품질 하락 억제

대회 채점은 **성능 지표와 속도 지표를 절반씩 반영**하는 구조이며, 본 실험은 두 지표를 함께 밀어 올리는 조합을 찾는 데 초점을 맞췄습니다.

---

## 🧠 방법 (GPTQ · Pruning · Distillation)

각 스크립트는 baseline에서 한 가지 가설을 바꿔 검증하는 단위 실험으로 구성했습니다.

### 1. GPTQ W4A16 양자화 (기반 기법)
`llmcompressor`의 `GPTQModifier`를 사용해 모든 `Linear` 레이어를 4비트로 양자화하고, `embed_tokens`·`lm_head`는 성능 보호를 위해 제외했습니다. 캘리브레이션은 `LGAI-EXAONE/MANTA-1M` 데이터셋(256~2048 샘플, `max_seq_length` 512)으로 수행하고, `dampening_frac=0.01`로 가중치 갱신을 안정화했습니다. (baseline 레시피 자체는 대회 제공)

### 2. 시드 고정 + 어텐션 q/k 제외 (`seed_equal.py`)
`random`·`numpy`·`torch`·CUDA 시드를 모두 `seed=1`로 고정하고 cuDNN을 deterministic 모드로 설정해 **재현 가능한 양자화 결과**를 확보했습니다. 여기에 `ignore` 목록에 `q_proj`·`k_proj`를 추가해, 어텐션의 query/key 경로를 양자화 대상에서 빼 정확도 영향을 관찰했습니다.

### 3. 채널 Pruning → Distillation → GPTQ (`scale_pruning_distill.py`)
3단계 파이프라인으로 구성했습니다.
- **Phase 1 — Scale Pruning**: MLP의 `proj` Linear 레이어에서 출력 채널별 L2 norm을 계산해 하위 50% 채널의 가중치를 0으로 만들어 제거
- **Phase 2 — Knowledge Distillation**: 원본 모델을 teacher로 두고 `KLDivLoss`(temperature 2.0, 1 epoch, lr 2e-5)로 pruning 손실을 복구하는 학습 수행 (`transformers.Trainer` 커스텀)
- **Phase 3 — GPTQ 4-bit**: pruning·증류를 거친 모델을 group-wise int4(`group_size=128`, symmetric)로 양자화하고, 결과를 `submit.zip`으로 패키징

### 4. Near-identity 레이어 무력화 (`nearly_zero.py`)
마지막 6개 레이어(24~29)의 `self_attn.o_proj`와 `mlp.down_proj` 가중치를 0으로 만들어, **residual 경로 덕분에 해당 블록이 항등 함수에 가깝게 동작**하도록 무력화한 뒤 전체를 GPTQ W4A16으로 양자화했습니다. 30 레이어 구조는 유지하면서 유효 연산량을 줄여 속도를 확보하려는 시도입니다.

### 5. 마지막 레이어 양자화 제외 (`ignore_dummy_GPTQ.py`)
사전 학습된 체크포인트(`trainer_output_v6_padded`)를 로드하고, 레이어 24~29의 모든 Linear를 GPTQ `ignore`에 넣어 **bf16 정밀도로 유지**(양자화 제외)했습니다. 대회 제출 환경의 30 레이어 구조에 맞추기 위한 접근입니다.

### 6. 서빙 호환 보정 (공통)
양자화 저장 후 safetensors 가중치 키의 `model.model.*` 중복 접두를 `model.*`로 정규화하는 `normalize_safetensors_keys`를 적용해, vLLM과 Transformers 간 로딩 호환성을 맞췄습니다.

> **참고 — 실험 과정 기록**: 레이어 수를 줄여 속도를 올리려 했으나, 제출 환경(30 레이어 고정)에 맞추려고 더미 레이어를 0으로 되채우는 방식은 부적절하다고 판단해 `lm_head` 중복 제거 방향으로 정리했습니다. 원본 README가 언급한 `awq_gptq.py`(AWQ + GPTQ 조합)는 **현재 저장소에 파일이 포함되어 있지 않아** 본 문서에서는 결과를 기술하지 않습니다.

---

## 🛠 기술 스택

| 구분 | 기술 |
|---|---|
| **베이스 모델** | `LGAI-EXAONE/EXAONE-4.0-1.2B` (30 layers, bfloat16) |
| **양자화** | GPTQ W4A16 (`llmcompressor` `GPTQModifier`), Marlin 커널 경로 목표 |
| **압축 기법** | 채널 L2 pruning(50%), near-identity 레이어 무력화, group-wise int4(group_size 128, symmetric) |
| **학습(증류)** | Knowledge Distillation (`KLDivLoss`, temperature 2.0, `transformers.Trainer`) |
| **캘리브레이션 데이터** | `LGAI-EXAONE/MANTA-1M` (256~2048 samples, max_seq 512) |
| **프레임워크** | PyTorch (bfloat16), Hugging Face Transformers · Datasets, llmcompressor |
| **서빙 호환** | vLLM (safetensors 키 정규화), `save_compressed=True` |
| **평가** | 자체 `estimate_score.py` (PerfNorm + SpeedNorm) |
| **실행 환경** | Google Colab (Drive 경로 기반), Python |

---

## 📁 프로젝트 구조

```
LG_Aimers_8th_Quant/            # Fork of panhong99/LG_Aimers
├── default.py / default.ipynb  # 대회 제공 baseline GPTQ W4A16 (본인 작성 아님)
├── seed_equal.py               # 시드 고정 + q_proj·k_proj 양자화 제외 실험
├── scale_pruning_distill.py    # 채널 pruning → KD 증류 → GPTQ 4bit 파이프라인
├── nearly_zero.py              # 마지막 6레이어 near-identity 무력화 후 GPTQ
├── ignore_dummy_GPTQ.py        # 마지막 6레이어 양자화 제외 + vLLM 키 정규화
├── estimate_score.py           # 로컬 점수 추정 하네스 (PerfNorm + SpeedNorm)
└── README.md
```

각 스크립트는 독립 실행형으로, 상단의 `MODEL_ID`·`OUT_DIR` 경로만 바꿔 개별 실험을 재현합니다.

---

## 🧪 실험 · 결과

> **정직성 고지**: 저장소에는 각 실험의 **최종 점수·로그가 커밋되어 있지 않습니다.** 아래는 코드에서 확인 가능한 실험 설계와 산출물만 정리한 것이며, 수치 성능은 별도로 기록되어 있지 않습니다. `estimate_score.py`의 baseline 상수(예: 토큰당 0.015초)는 코드에 `예시(placeholder)`로 표기되어 있어 절대 성능치로 해석하지 않습니다.

### 수행한 단위 실험

| 스크립트 | 바꾼 변수 | 산출물 |
|---|---|---|
| `seed_equal.py` | 시드 고정(재현성) + q/k_proj 양자화 제외 | W4A16 양자화 모델 |
| `scale_pruning_distill.py` | MLP 채널 50% pruning + 증류 복구 + int4 | 양자화 모델 + `submit.zip` |
| `nearly_zero.py` | 마지막 6레이어 무력화(항등화) + 전체 양자화 | vLLM 호환 양자화 모델 |
| `ignore_dummy_GPTQ.py` | 마지막 6레이어 bf16 유지(양자화 제외) | vLLM 호환 양자화 모델 |

### 자체 채점 방식 (`estimate_score.py`)

대회 채점식을 로컬에서 근사하기 위해 다음을 구현했습니다.

- **PerfNorm(성능)**: 양자화 모델의 생성문과 정답 문장의 단어 overlap 기반 F1을 baseline 대비 정규화
- **SpeedNorm(속도)**: 토큰당 생성 시간을 baseline 대비 정규화 (더 빠를수록 높은 점수)
- **최종 점수**: `Score = 0.5 × PerfNorm + 0.5 × SpeedNorm`
- **Quick 모드**: 저장된 모델 파일 크기의 압축률만으로 점수를 빠르게 근사

이 하네스는 실측 데이터셋(MANTA-1M 후미 100개)으로 성능을, 20개 샘플로 속도를 측정하도록 작성되어 있어, 모델 디렉터리만 넘기면 성능·속도·압축률을 한 번에 추정합니다.

---

## 🐛 로컬 실행

```bash
# 의존성 설치 (Colab/로컬 공통)
pip install torch transformers datasets llmcompressor safetensors

# 실험 실행 전, 각 스크립트 상단의 MODEL_ID / OUT_DIR 경로를 환경에 맞게 수정

# 예: pruning → distillation → GPTQ 파이프라인
python scale_pruning_distill.py

# 예: 마지막 6레이어 무력화 후 양자화
python nearly_zero.py

# 양자화 결과 점수 추정
python estimate_score.py ./model_exp25          # 전체 평가
python estimate_score.py ./model_exp25 --quick  # 크기 기반 빠른 추정
```

베이스 모델(`LGAI-EXAONE/EXAONE-4.0-1.2B`)과 캘리브레이션 데이터셋(`LGAI-EXAONE/MANTA-1M`)은 Hugging Face에서 로드하며, 스크립트 다수가 Google Colab의 Drive 경로(`/content/drive/...`)를 기준으로 작성되어 있어 실행 환경에 맞게 경로를 조정합니다.

---

## 📌 정리

이 저장소는 **LG Aimers 8기 LLM Compression 트랙**에서 EXAONE-4.0-1.2B를 대상으로 진행한 양자화 경량화 실험(Fork)입니다. GPTQ W4A16을 기반으로 시드 고정, 채널 pruning + 증류, 레이어 무력화, 부분 양자화 제외를 각각 스크립트로 분리해 검증하고, 대회 채점식을 근사하는 자체 평가 하네스까지 갖췄습니다. 본인 기여분은 위 실험 스크립트와 평가 코드이며, baseline(`default.*`)은 대회 제공 코드입니다.
