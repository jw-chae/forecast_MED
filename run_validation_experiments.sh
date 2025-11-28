#!/bin/bash
# Run validation experiments for all three models

cd /home/joongwon00/Project_Tsinghua_Paper/med_deepseek
source ~/anaconda3/bin/activate epillm

echo "=========================================="
echo "Starting Validation Experiments (Round 2)"
echo "=========================================="

# OpenAI GPT-5.1 - Round 2
echo ""
echo "[1/4] Starting OpenAI GPT-5.1 validation..."
nohup python -m experiments.core.rolling_agent_forecast \
  --disease 手足口病 \
  --start 2024-02-01 \
  --end 2024-09-30 \
  --n_steps 0 \
  --horizon 1 \
  --model gpt-5.1 \
  --provider openai \
  --temperature 0.6 \
  --save_json \
  --batch hfmd_openai_gpt51_v2 \
  > logs/openai_gpt51_v2.log 2>&1 &

echo "PID: $!"
sleep 2

# Google Gemini - Round 2
echo ""
echo "[2/4] Starting Google Gemini validation..."
nohup python -m experiments.core.rolling_agent_forecast \
  --disease 手足口病 \
  --start 2024-02-01 \
  --end 2024-09-30 \
  --n_steps 0 \
  --horizon 1 \
  --model gemini-3-pro-preview \
  --provider google \
  --temperature 0.6 \
  --save_json \
  --batch hfmd_gemini_3pro_v2 \
  > logs/gemini_3pro_v2.log 2>&1 &

echo "PID: $!"
sleep 2

# DeepSeek Reasoner - Round 2
echo ""
echo "[3/4] Starting DeepSeek Reasoner validation..."
nohup python -m experiments.core.rolling_agent_forecast \
  --disease 手足口病 \
  --start 2024-02-01 \
  --end 2024-09-30 \
  --n_steps 0 \
  --horizon 1 \
  --model deepseek-reasoner \
  --provider deepseek \
  --temperature 0.6 \
  --save_json \
  --batch hfmd_deepseek_reasoner_v2 \
  > logs/deepseek_reasoner_v2.log 2>&1 &

echo "PID: $!"
sleep 2

# Hong Kong Data with DeepSeek (best performer)
echo ""
echo "[4/4] Starting Hong Kong HFMD experiment with DeepSeek..."
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
echo "All experiments started!"
echo "=========================================="
echo ""
echo "Monitor progress with:"
echo "  ps aux | grep rolling_agent_forecast | grep -v grep"
echo ""
echo "Check logs:"
echo "  tail -f logs/openai_gpt51_v2.log"
echo "  tail -f logs/gemini_3pro_v2.log"
echo "  tail -f logs/deepseek_reasoner_v2.log"
echo "  tail -f logs/hongkong_deepseek.log"
