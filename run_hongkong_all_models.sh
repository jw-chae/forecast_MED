#!/bin/bash
# Run Hong Kong HFMD experiments with all four LLM providers

cd /home/joongwon00/Project_Tsinghua_Paper/med_deepseek
source ~/anaconda3/bin/activate epillm

echo "=========================================="
echo "Starting Hong Kong HFMD Experiments"
echo "Data: 2010-2025 (829 weeks)"
echo "Period: 2023-01-01 to 2024-09-30"
echo "=========================================="

# 1. Qwen (DashScope)
echo ""
echo "[1/4] Starting Qwen3-235b-a22b (DashScope)..."
nohup python -m experiments.core.rolling_agent_forecast \
  --disease 手足口病 \
  --csv_path experiments/data_for_model/手足口病/data_HK/hk_hfmd_weekly_2010_2025.csv \
  --start 2023-01-01 \
  --end 2024-09-30 \
  --n_steps 0 \
  --horizon 1 \
  --model qwen-plus \
  --provider dashscope \
  --temperature 0.6 \
  --save_json \
  --batch hfmd_hongkong_qwen \
  > logs/hongkong_qwen.log 2>&1 &

echo "PID: $!"
sleep 2

# 2. OpenAI GPT-5.1
echo ""
echo "[2/4] Starting OpenAI GPT-5.1..."
nohup python -m experiments.core.rolling_agent_forecast \
  --disease 手足口病 \
  --csv_path experiments/data_for_model/手足口病/data_HK/hk_hfmd_weekly_2010_2025.csv \
  --start 2023-01-01 \
  --end 2024-09-30 \
  --n_steps 0 \
  --horizon 1 \
  --model gpt-5.1 \
  --provider openai \
  --temperature 0.6 \
  --save_json \
  --batch hfmd_hongkong_openai \
  > logs/hongkong_openai.log 2>&1 &

echo "PID: $!"
sleep 2

# 3. Google Gemini 3 Pro
echo ""
echo "[3/4] Starting Google Gemini 3 Pro..."
nohup python -m experiments.core.rolling_agent_forecast \
  --disease 手足口病 \
  --csv_path experiments/data_for_model/手足口病/data_HK/hk_hfmd_weekly_2010_2025.csv \
  --start 2023-01-01 \
  --end 2024-09-30 \
  --n_steps 0 \
  --horizon 1 \
  --model gemini-3-pro-preview \
  --provider google \
  --temperature 0.6 \
  --save_json \
  --batch hfmd_hongkong_gemini \
  > logs/hongkong_gemini.log 2>&1 &

echo "PID: $!"
sleep 2

# 4. DeepSeek Reasoner
echo ""
echo "[4/4] Starting DeepSeek Reasoner..."
nohup python -m experiments.core.rolling_agent_forecast \
  --disease 手足口病 \
  --csv_path experiments/data_for_model/手足口病/data_HK/hk_hfmd_weekly_2010_2025.csv \
  --start 2023-01-01 \
  --end 2024-09-30 \
  --n_steps 0 \
  --horizon 1 \
  --model deepseek-reasoner \
  --provider deepseek \
  --temperature 0.6 \
  --save_json \
  --batch hfmd_hongkong_deepseek \
  > logs/hongkong_deepseek.log 2>&1 &

echo "PID: $!"

echo ""
echo "=========================================="
echo "All Hong Kong experiments started!"
echo "=========================================="
echo ""
echo "Monitor progress with:"
echo "  ps aux | grep rolling_agent_forecast | grep -v grep"
echo ""
echo "Check logs:"
echo "  tail -f logs/hongkong_qwen.log"
echo "  tail -f logs/hongkong_openai.log"
echo "  tail -f logs/hongkong_gemini.log"
echo "  tail -f logs/hongkong_deepseek.log"
