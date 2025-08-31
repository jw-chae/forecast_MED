# 아이디어 1: 시계열 예측 모델 기반 중증 악화 예측 및 MLLM 에이전트 개발

## 완료된 작업 (Done)

1.  **데이터 전처리 파이프라인 구축**
    *   5개의 원본 Excel 파일(HIS, LIS, PACS, EMR 등)을 통합하여 `analysis_ready_data.json` 데이터셋 생성 완료.
    *   환자 ID 기준 데이터 분할 (학습 70%, 검증 15%, 테스트 15%) 완료 및 `train/validation/test_dataset.json` 파일 생성.

2.  **예측 모델링 (1단계)**
    *   **실제 Outcome 생성**: ICU 입실 또는 14일 내 재입원 기준의 실제 중증 악화 Y 레이블(`outcomes.json`) 생성 완료.
    *   **시계열 데이터 변환**: 딥러닝 모델 학습을 위해 72시간 시계열 데이터를 포함하는 `_timeseries.parquet` 파일 생성 완료.
    *   **딥러닝 모델 학습**:
        *   `pytorch-forecasting` 라이브러리의 버전 및 환경 문제 해결.
        *   `TemporalFusionTransformer` 모델 학습을 완료하고 모델(`tft_model.pt`)과 데이터셋(`tft_training.tsd`) 아티팩트 저장 완료.

## 해야 할 일 (To-Do)

1.  **TFT 모델 성능 평가**
    *   저장된 모델과 데이터셋을 로드하여 테스트 데이터셋에 대한 예측 수행.
    *   주요 성능 지표 계산:
        *   AUROC (Area Under the ROC Curve)
        *   AUPRC (Area Under the Precision-Recall Curve)
        *   Classification Report (Precision, Recall, F1-score)
    *   모델 신뢰도 평가를 위한 Calibration Plot 생성 및 `reports/figures/`에 저장.

2.  **MLLM 에이전트 개발 (초기 단계)**
    *   SFT(Supervised Fine-Tuning) 데이터셋 생성을 위한 `scripts/preprocessing/04_create_sft_dataset.py` 스크립트 작성.
    *   스크립트는 `train_dataset.json`을 로드하여 각 환자 데이터에 대해 LLM (e.g., Gemini 1.5 Pro)을 호출.
    *   LLM을 통해 환자 상태 요약, 사고 과정(thinking_process), 추천 오더(suggested_orders)가 포함된 초안 생성.
    *   생성된 초안을 `processed_data/sft_draft_dataset.json` 파일로 저장.

3.  **모델 앙상블 및 보정 (Ensemble & Calibration)**
    *   (필요시) 로지스틱 회귀 모델과 TFT 모델의 예측 결과를 앙상블하여 최종 예측 점수 산출.
    *   검증 데이터셋을 이용해 Platt 보정 또는 Isotonic 회귀를 수행하여 모델의 신뢰도 보정.
