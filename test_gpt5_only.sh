#!/bin/bash
START_DATE="2024-09-01"
END_DATE="2024-09-15"
HORIZON=1
N_STEPS=1

echo "========================================================"
echo "Testing GPT-5.1 (gpt-5.1)"
echo "========================================================"
python -m experiments.core.rolling_agent_forecast \
    --disease "手足口病" \
    --start $START_DATE \
    --end $END_DATE \
    --horizon $HORIZON \
    --n_steps $N_STEPS \
    --model "gpt-5.1" \
    --provider "openai" \
    --forecast_mode "advanced" \
    --batch "test_gpt5_short_retry"
