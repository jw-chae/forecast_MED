#!/bin/bash

# Baseline experiments: Model2 ONLY (Forecast Generator) with LLM
# Model1 (Event Interpreter) is skipped, using neutral values (impact=0, confidence=0.5)
# 3 model types × 2 datasets = 6 experiments

BASE_DIR="/home/joongwon00/Project_Tsinghua_Paper/med_deepseek"
cd "$BASE_DIR" || exit 1
source ~/anaconda3/bin/activate epillm

# Hangzhou hospital dataset (3 experiments)
echo "Starting Hangzhou baseline experiments (Model2 only)..."

# 1. Qwen baseline - Hangzhou
nohup python -m experiments.core.rolling_agent_forecast \
  --disease 手足口病 \
  --start 2024-02-01 \
  --end 2024-09-30 \
  --n_steps 0 \
  --horizon 1 \
  --model qwen3-235b-a22b \
  --provider dashscope \
  --temperature 0.6 \
  --skip_model1 \
  --save_json \
  --batch baseline_hangzhou_qwen \
  > logs/baseline_hangzhou_qwen.log 2>&1 &
echo "Hangzhou + Qwen baseline started (PID: $!)"

# 2. OpenAI baseline - Hangzhou
nohup python -m experiments.core.rolling_agent_forecast \
  --disease 手足口病 \
  --start 2024-02-01 \
  --end 2024-09-30 \
  --n_steps 0 \
  --horizon 1 \
  --model gpt-5.1 \
  --provider openai \
  --temperature 1.0 \
  --skip_model1 \
  --save_json \
  --batch baseline_hangzhou_openai \
  > logs/baseline_hangzhou_openai.log 2>&1 &
echo "Hangzhou + OpenAI baseline started (PID: $!)"

# 3. DeepSeek baseline - Hangzhou
nohup python -m experiments.core.rolling_agent_forecast \
  --disease 手足口病 \
  --start 2024-02-01 \
  --end 2024-09-30 \
  --n_steps 0 \
  --horizon 1 \
  --model deepseek-reasoner \
  --provider deepseek \
  --temperature 1.0 \
  --skip_model1 \
  --save_json \
  --batch baseline_hangzhou_deepseek \
  > logs/baseline_hangzhou_deepseek.log 2>&1 &
echo "Hangzhou + DeepSeek baseline started (PID: $!)"

# Hong Kong dataset (3 experiments)
echo "Starting Hong Kong baseline experiments (Model2 only)..."

# 4. Qwen baseline - Hong Kong
nohup python -m experiments.core.rolling_agent_forecast \
  --disease 手足口病 \
  --csv_path experiments/data_for_model/手足口病/data_HK/hk_hfmd_weekly_2010_2025.csv \
  --start 2023-01-01 \
  --end 2024-09-30 \
  --n_steps 0 \
  --horizon 1 \
  --model qwen3-235b-a22b \
  --provider dashscope \
  --temperature 1.0 \
  --skip_model1 \
  --save_json \
  --batch baseline_hongkong_qwen \
  > logs/baseline_hongkong_qwen.log 2>&1 &
echo "Hong Kong + Qwen baseline started (PID: $!)"

# 5. OpenAI baseline - Hong Kong
nohup python -m experiments.core.rolling_agent_forecast \
  --disease 手足口病 \
  --csv_path experiments/data_for_model/手足口病/data_HK/hk_hfmd_weekly_2010_2025.csv \
  --start 2023-01-01 \
  --end 2024-09-30 \
  --n_steps 0 \
  --horizon 1 \
  --model gpt-5.1 \
  --provider openai \
  --temperature 1.0 \
  --skip_model1 \
  --save_json \
  --batch baseline_hongkong_openai \
  > logs/baseline_hongkong_openai.log 2>&1 &
echo "Hong Kong + OpenAI baseline started (PID: $!)"

# 6. DeepSeek baseline - Hong Kong
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
  --skip_model1 \
  --save_json \
  --batch baseline_hongkong_deepseek \
  > logs/baseline_hongkong_deepseek.log 2>&1 &
echo "Hong Kong + DeepSeek baseline started (PID: $!)"

echo ""
echo "========================================="
echo "All 6 baseline experiments started!"
echo "========================================="
echo "3 Hangzhou experiments (2024-02-01 to 2024-09-30)"
echo "3 Hong Kong experiments (2023-01-01 to 2024-09-30)"
echo ""
echo "Using --skip_model1 flag:"
echo "  - Model1 (EventInterpreter): SKIPPED (neutral: impact=0, confidence=0.5)"
echo "  - Model2 (ForecastGenerator): LLM enabled"
echo "========================================="
