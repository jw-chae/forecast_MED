#!/bin/bash

# Test multiple models with a short 1-week forecast
# Using a fixed HFMD date range in early July 2024

START_DATE="2024-07-01"
END_DATE="2024-07-15"
HORIZON=1
N_STEPS=1

echo "========================================================"
echo "Testing Qwen (qwen3-235b-a22b, T=0.6, thinking)"
echo "========================================================"
python -m experiments.core.rolling_agent_forecast \
    --disease "手足口病" \
    --start $START_DATE \
    --end $END_DATE \
    --horizon $HORIZON \
    --n_steps $N_STEPS \
    --model "qwen3-235b-a22b" \
    --provider "dashscope" \
    --temperature 0.6 \
    --forecast_mode "advanced" \
    --batch "test_qwen_short" \
    --save_json

echo ""
echo "========================================================"
echo "Testing DeepSeek (deepseek-reasoner, T=1.0)"
echo "========================================================"
python -m experiments.core.rolling_agent_forecast \
    --disease "手足口病" \
    --start $START_DATE \
    --end $END_DATE \
    --horizon $HORIZON \
    --n_steps $N_STEPS \
    --model "deepseek-reasoner" \
    --provider "deepseek" \
    --temperature 1.0 \
    --forecast_mode "advanced" \
    --batch "test_deepseek_short" \
    --save_json

echo ""
echo "========================================================"
echo "Testing GPT-5.1 (gpt-5.1, T=0.2)"
echo "========================================================"
python -m experiments.core.rolling_agent_forecast \
    --disease "手足口病" \
    --start $START_DATE \
    --end $END_DATE \
    --horizon $HORIZON \
    --n_steps $N_STEPS \
    --model "gpt-5.1" \
    --provider "openai" \
    --temperature 1.0 \
    --forecast_mode "advanced" \
    --batch "test_gpt5_short" \
    --save_json

echo ""
echo "========================================================"
echo "Test Complete (Qwen + DeepSeek + GPT-5.1)"
echo "========================================================"
