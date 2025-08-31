# Med-DeepSeek 시스템 사용법 및 결과

이 문서는 Med-DeepSeek 예측 시스템의 주요 스크립트 사용법과 실행 결과를 정리합니다.

## 1. 주요 스크립트 사용법

`Project_Tsinghua_Paper/med_deepseek/scripts/` 디렉토리에서 아래 명령어들을 실행합니다.

### 1.1. 단일 미래 예측 및 시각화

현재 시점에서 미래를 빠르게 예측하고 HTML 리포트로 결과를 확인하고 싶을 때 사용합니다.

-   **스크립트:** `Tools/generate_html_report.py`
-   **설명:** 사용 가능한 모든 과거 데이터를 사용하여 지정된 기간(`horizon`) 만큼 미래를 예측하고, 인터랙티브 차트가 포함된 HTML 파일을 생성합니다.
-   **주요 인자:**
    -   `--disease`: 예측할 질병명 (예: "流行性感冒")
    -   `--horizon`: 예측할 기간 (주 단위, 예: 26)
-   **실행 예시:**
    ```bash
    python3 -m Tools.generate_html_report --disease "流行性感冒" --horizon 26
    ```
-   **결과:** `reports/` 디렉토리에 `forecast_流行性感冒_....html`와 같은 이름의 파일이 생성됩니다.

### 1.2. 롤링 예측 백테스트

과거 여러 시점에서 모델의 예측 성능을 종합적으로 평가(백테스트)할 때 사용합니다.

-   **스크립트:** `Tools/rolling_agent_forecast.py`
-   **설명:** 지정된 기간 동안 예측 시점을 한 주씩 이동해가며 LLM 에이전트 기반의 예측을 반복 수행하고, 전체 결과를 종합한 HTML 리포트를 생성합니다.
-   **주요 인자:**
    -   `--disease`: 예측할 질병명
    -   `--start`: 백테스트 시작일 (예: "2023-01-01")
    -   `--end`: 백테스트 종료일 (예: "2023-12-31")
    -   `--provider`: 사용할 LLM 프로바이더 (예: "openai", "dashscope")
-   **실행 예시:**
    ```bash
    python3 -m Tools.rolling_agent_forecast --disease "手足口病" --start "2023-01-01" --end "2023-12-31" --provider openai
    ```
-   **결과:** `reports/rolling_forecasts/` 디렉토리에 HTML 리포트가 생성됩니다.

### 1.3. LLM 미세조정(Fine-tuning) 파이프라인

LLM 에이전트의 성능을 개선하기 위해 오프라인 데이터셋을 구축하고, 이를 이용해 모델을 미세조정합니다.

**1단계: 데이터셋 생성**

-   **스크립트:** `Tools/offline_dataset_builder.py`
-   **설명:** 지정된 기간 동안 (상태, 행동, 보상) 튜플을 수집하여 오프라인 강화학습 또는 모방학습을 위한 데이터셋(JSONL 형식)을 생성합니다.
-   **실행 예시:**
    ```bash
    python3 -m Tools.offline_dataset_builder --disease "手足口病" --start "2023-01-01" --end "2023-12-31" --provider openai
    ```
-   **결과:** `reports/offline_dataset/` 디렉토리에 `ppo_dataset_...jsonl` 파일이 생성됩니다.

**2단계: 미세조정 실행**

-   **스크립트:** `Tools/finetune_agent_llm.py`
-   **설명:** 1단계에서 생성한 데이터셋을 사용하여 사전 학습된 언어 모델(예: Qwen)을 미세조정합니다.
-   **실행 예시:**
    ```bash
    python3 -m Tools.finetune_agent_llm --dataset_path reports/offline_dataset/ppo_dataset_手足口病_....jsonl --output_dir ./finetuned_agent
    ```
-   **결과:** `--output_dir`로 지정한 디렉토리에 학습된 모델 어댑터(LoRA 가중치)가 저장됩니다.

## 2. 실행 결과

### 2.1. 2024년 1-6월 인플루엔자 예측 (Holdout Validation)

-   **실행 일시:** 2024-08-31
-   **명령어:** `python3 -m Tools.holdout_forecast_html --disease "流行性感冒" --train_until "2023-12-31" --end "2024-06-30"`
-   **결과 파일:** `/home/joongwon00/Project_Tsinghua_Paper/med_deepseek/reports/holdout_forecast_流行性感冒_2023-12-31_2024-06-30.html`
-   **요약:** 2023년 말까지의 데이터를 사용하여 2024년 상반기 인플루엔자 유행을 예측하고, 이를 해당 기간의 실제 데이터와 비교하여 모델의 성능을 검증했습니다.
