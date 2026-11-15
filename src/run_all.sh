#!/bin/bash
# Runs Strategy A and B for both models on both team datasets, one after another, on ONE GPU.
# Both models in 16-bit (no 8-bit), same settings everywhere.
#
#   cd ~/Team-Horizon
#   nohup bash src/run_all.sh > run_all.log 2>&1 &
#   tail -f run_all.log
#
# If the server restarts: start it again with  RESUME=--resume nohup bash src/run_all.sh > run_all.log 2>&1 &
# (finished tasks are skipped).
#
# GPU choice (check nvidia-smi first):  GPU = GPU for Qwen (default 2),  GPU_D = GPU(s) for DeepSeek (default = GPU).
# DeepSeek in 16-bit needs ~30 GB. If one GPU doesn't have that free, split it over two:  GPU_D=2,1
#   GPU=2 GPU_D=2,1 nohup bash src/run_all.sh > run_all.log 2>&1 &
set -e
cd ~/Team-Horizon
GPU=${GPU:-2}
GPU_D=${GPU_D:-$GPU}
RESUME=${RESUME:-}

Q=Qwen/Qwen2.5-Coder-7B-Instruct
D=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
PY_Q=python                    # Qwen: normal environment (as before)
PY_D=$HOME/venv-glm/bin/python # DeepSeek: venv-glm (transformers version that works with DeepSeek)
COMMON="--dataset mbpp --max-new-tokens 512 $RESUME"

echo "===== Qwen on GPU $GPU, DeepSeek on GPU $GPU_D ===="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

for DATA in final_30 initial_10; do
  F=Dataset/mbpp_$DATA.jsonl
  echo "===== $(date '+%H:%M') Qwen A  $DATA ====="
  CUDA_VISIBLE_DEVICES=$GPU $PY_Q -u src/baseline_oneshot.py --backend hf --model $Q --data-file $F $COMMON
  echo "===== $(date '+%H:%M') Qwen B  $DATA ====="
  CUDA_VISIBLE_DEVICES=$GPU $PY_Q -u src/strategy_b.py --backend hf --model $Q --data-file $F $COMMON
  echo "===== $(date '+%H:%M') DeepSeek A  $DATA ====="
  CUDA_VISIBLE_DEVICES=$GPU_D $PY_D -u src/baseline_oneshot.py --backend hf --model $D --no-remote-code --data-file $F $COMMON
  echo "===== $(date '+%H:%M') DeepSeek B  $DATA ====="
  CUDA_VISIBLE_DEVICES=$GPU_D $PY_D -u src/strategy_b.py --backend hf --model $D --no-remote-code --data-file $F $COMMON
done
echo "===== $(date '+%H:%M') ALL DONE ====="
