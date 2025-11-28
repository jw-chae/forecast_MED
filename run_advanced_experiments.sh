#!/bin/bash

# Advanced mode experiments (Model 1 + Model 2 with direct prediction)
# 4 model types × 2 datasets = 8 experiments

BASE_DIR="/home/joongwon00/Project_Tsinghua_Paper/med_deepseek"
cd "$BASE_DIR" || exit 1
source ~/anaconda3/bin/activate epillm

echo "========================================="
echo "Starting Advanced Mode Experiments"
echo "Model 1 + Model 2 (forecast_mode=advanced)"
echo "========================================="

# Hangzhou hospital dataset (4 experiments)
echo ""
echo "Starting Hangzhou advanced experiments..."

# 1. Qwen advanced - Hangzhou
nohup python -m experiments.core.rolling_agent_forecast \
  --disease 手足口病 \
  --start 2024-02-01 \
  --end 2024-09-30 \
  --n_steps 0 \
  --horizon 1 \
  --model qwen3-235b-a22b \
  --provider dashscope \
  --temperature 0.6 \
  --forecast_mode advanced \
  --save_json \
  --batch advanced_hangzhou_qwen \
  > logs/advanced_hangzhou_qwen.log 2>&1 &
echo "Hangzhou + Qwen advanced started (PID: $!)"

# 2. OpenAI advanced - Hangzhou
nohup python -m experiments.core.rolling_agent_forecast \
  --disease 手足口病 \
  --start 2024-02-01 \
  --end 2024-09-30 \
  --n_steps 0 \
  --horizon 1 \
  --model gpt-5.1 \
  --provider openai \
  --temperature 0.2 \
  --forecast_mode advanced \
  --save_json \
  --batch advanced_hangzhou_openai \
  > logs/advanced_hangzhou_openai.log 2>&1 &
echo "Hangzhou + OpenAI advanced started (PID: $!)"

# 3. DeepSeek advanced - Hangzhou
nohup python -m experiments.core.rolling_agent_forecast \
  --disease 手足口病 \
  --start 2024-02-01 \
  --end 2024-09-30 \
  --n_steps 0 \
  --horizon 1 \
  --model deepseek-reasoner \
  --provider deepseek \
  --temperature 1.0 \
  --forecast_mode advanced \
  --save_json \
  --batch advanced_hangzhou_deepseek \
  > logs/advanced_hangzhou_deepseek.log 2>&1 &
echo "Hangzhou + DeepSeek advanced started (PID: $!)"

# Hong Kong dataset (4 experiments)
echo ""
echo "Starting Hong Kong advanced experiments..."

# 5. Qwen advanced - Hong Kong
nohup python -m experiments.core.rolling_agent_forecast \
  --disease 手足口病 \
  --csv_path experiments/data_for_model/手足口病/data_HK/hk_hfmd_weekly_2010_2025.csv \
  --start 2023-01-01 \
  --end 2024-09-30 \
  --n_steps 0 \
  --horizon 1 \
  --model qwen3-235b-a22b \
  --provider dashscope \
  --temperature 0.6 \
  --forecast_mode advanced \
  --save_json \
  --batch advanced_hongkong_qwen \
  > logs/advanced_hongkong_qwen.log 2>&1 &
echo "Hong Kong + Qwen advanced started (PID: $!)"

# 6. OpenAI advanced - Hong Kong
nohup python -m experiments.core.rolling_agent_forecast \
  --disease 手足口病 \
  --csv_path experiments/data_for_model/手足口病/data_HK/hk_hfmd_weekly_2010_2025.csv \
  --start 2023-01-01 \
  --end 2024-09-30 \
  --n_steps 0 \
  --horizon 1 \
  --model gpt-5.1 \
  --provider openai \
  --temperature 0.2 \
  --forecast_mode advanced \
  --save_json \
  --batch advanced_hongkong_openai \
  > logs/advanced_hongkong_openai.log 2>&1 &
echo "Hong Kong + OpenAI advanced started (PID: $!)"

# 7. DeepSeek advanced - Hong Kong
nohup python -m experiments.core.rolling_agent_forecast \
  --disease 手足口病 \
  --csv_path experiments/data_for_model/手足口病/data_HK/hk_hfmd_weekly_2010_2025.csv \
  --start 2023-01-01 \
  --end 2024-09-30 \
  --n_steps 0 \
  --horizon 1 \
  --model deepseek-reasoner \
  --provider deepseek \
  --temperature 1.0 \
  --forecast_mode advanced \
  --save_json \
  --batch advanced_hongkong_deepseek \
  > logs/advanced_hongkong_deepseek.log 2>&1 &
echo "Hong Kong + DeepSeek advanced started (PID: $!)"

echo ""
echo "========================================="
echo "All 6 advanced mode experiments started!"
echo "========================================="
echo "3 Hangzhou experiments (2024-02-01 to 2024-09-30)"
echo "3 Hong Kong experiments (2023-01-01 to 2024-09-30)"
echo ""
echo "Using --forecast_mode advanced"
echo "Model 1 (Event Interpreter) + Model 2 (Direct Prediction)"
echo "========================================="
