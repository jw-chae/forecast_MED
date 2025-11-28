#!/bin/bash
# HZ HFMD Baseline 실험 실행 스크립트
# 데이터: processed_data/lis_weekly_epi_top20_plus4_unique_counts.csv
# 질병: 手足口病
# 예측기간: 2024-02-01 ~ 2024-09-30
# Horizon: 1주

set -e  # 에러 발생시 중단

cd /home/joongwon00/Project_Tsinghua_Paper/med_deepseek

BATCH_NAME="hz_hfmd_baselines"
CONFIG_DIR="experiments/configs/baselines/hz_hfmd"

echo "=========================================="
echo "HZ HFMD Baseline 실험 시작"
echo "Batch: $BATCH_NAME"
echo "=========================================="

# ARIMA
echo "[1/7] ARIMA 실행중..."
python -m experiments.run_experiment \
  --config $CONFIG_DIR/arima.yaml \
  --batch $BATCH_NAME

# Prophet
echo "[2/7] Prophet 실행중..."
python -m experiments.run_experiment \
  --config $CONFIG_DIR/prophet.yaml \
  --batch $BATCH_NAME

# LSTM
echo "[3/7] LSTM 실행중..."
python -m experiments.run_experiment \
  --config $CONFIG_DIR/lstm.yaml \
  --batch $BATCH_NAME

# XGBoost
echo "[4/7] XGBoost 실행중..."
python -m experiments.run_experiment \
  --config $CONFIG_DIR/xgboost.yaml \
  --batch $BATCH_NAME

# Chronos
echo "[5/7] Chronos 실행중..."
python -m experiments.run_experiment \
  --config $CONFIG_DIR/chronos.yaml \
  --batch $BATCH_NAME

# Moirai
echo "[6/7] Moirai 실행중..."
python -m experiments.run_experiment \
  --config $CONFIG_DIR/moirai.yaml \
  --batch $BATCH_NAME

# TimesFM
echo "[7/7] TimesFM 실행중..."
python -m experiments.run_experiment \
  --config $CONFIG_DIR/timesfm.yaml \
  --batch $BATCH_NAME

echo "=========================================="
echo "HZ HFMD Baseline 실험 완료!"
echo "결과 위치: experiments/results/$BATCH_NAME/"
echo "=========================================="
