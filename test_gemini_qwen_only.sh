#!/bin/bash

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
    --batch "test_qwen_short"

echo ""
echo "========================================================"
echo "Testing DeepSeek (deepseek-reasoner, T=0.6)"
echo "========================================================"
python -m experiments.core.rolling_agent_forecast \
    --disease "手足口病" \
    --start $START_DATE \
    --end $END_DATE \
    --horizon $HORIZON \
    --n_steps $N_STEPS \
    --model "deepseek-reasoner" \
    --provider "deepseek" \
    --temperature 0.6 \
    --forecast_mode "advanced" \
    --batch "test_deepseek_short"

echo ""
echo "========================================================"
echo "Test Complete (Qwen3 + DeepSeek)"
echo "========================================================"

echo ""
echo "Saved artifacts overview:"
echo "  - JSON results:        results_json/test_qwen_short/*.json"
echo "  - Logs:                logs/test_qwen_short/*.log"
echo "  - Postprocessed (Qwen): experiments/results/test_qwen_short_*/*"
echo "  - JSON results:        results_json/test_deepseek_short/*.json"
echo "  - Logs:                logs/test_deepseek_short/*.log"
echo "  - Postprocessed (DeepSeek): experiments/results/test_deepseek_short_*/*"
