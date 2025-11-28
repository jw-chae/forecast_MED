#!/bin/bash

# Test advanced mode with DeepSeek only (1 week test)

BASE_DIR="/home/joongwon00/Project_Tsinghua_Paper/med_deepseek"
cd "$BASE_DIR" || exit 1
source ~/anaconda3/bin/activate epillm

echo "========================================="
echo "Testing Advanced Mode - DeepSeek Only"
echo "1 week test (2024-02-01 to 2024-02-12)"
echo "========================================="

python -m experiments.core.rolling_agent_forecast \
  --disease 手足口病 \
  --start 2024-02-01 \
  --end 2024-02-12 \
  --n_steps 1 \
  --horizon 1 \
  --model deepseek-reasoner \
  --provider deepseek \
  --temperature 0.6 \
  --forecast_mode advanced \
  --save_json \
  --batch test_advanced_deepseek

echo ""
echo "========================================="
echo "Test completed!"
echo "Check: logs/rolling_stdout.log"
echo "========================================="
