#!/bin/bash
# Runs A and B (1 + 5 repairs) for both models on both datasets, then C (agent, 24 steps) on final_30.
# At the end it runs the checks and writes the failure reports to results/analysis/.
#
#   cd ~/Team-Horizon
#   GPU=1 GPU_D=0 nohup bash src/run_all.sh > run_all.log 2>&1 &
#   tail -f run_all.log
#
# GPU = GPU for Qwen, GPU_D = GPU for DeepSeek (needs about 30 GB free, check nvidia-smi first).
# If it stops halfway, start it again with RESUME=--resume (finished tasks are skipped).
set -e
cd ~/Team-Horizon
GPU=${GPU:-2}
GPU_D=${GPU_D:-$GPU}
RESUME=${RESUME:-}
REPAIRS=${REPAIRS:-5}   # Strategy B: repairs after the first try (5 -> at most 6 LLM calls)
STEPS=${STEPS:-24}      # Strategy C: max agent steps
ONLY_C=${ONLY_C:-}      # ONLY_C=1: skip A and B, run only C

Q=Qwen/Qwen2.5-Coder-7B-Instruct
D=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
PY_Q=python                    # Qwen: normal environment (as before)
PY_D=$HOME/venv-glm/bin/python # DeepSeek: venv-glm (transformers version that works with DeepSeek)
COMMON="--dataset mbpp --max-new-tokens 512 $RESUME"

echo "===== Qwen on GPU $GPU, DeepSeek on GPU $GPU_D ===="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

for DATA in final_30 initial_10; do
  [ -n "$ONLY_C" ] && break
  F=Dataset/mbpp_$DATA.jsonl
  echo "===== $(date '+%H:%M') Qwen A  $DATA ====="
  CUDA_VISIBLE_DEVICES=$GPU $PY_Q -u src/baseline_oneshot.py --backend hf --model $Q --data-file $F $COMMON
  echo "===== $(date '+%H:%M') Qwen B  $DATA ====="
  CUDA_VISIBLE_DEVICES=$GPU $PY_Q -u src/strategy_b.py --backend hf --model $Q --data-file $F --max-repairs $REPAIRS $COMMON
  echo "===== $(date '+%H:%M') DeepSeek A  $DATA ====="
  CUDA_VISIBLE_DEVICES=$GPU_D $PY_D -u src/baseline_oneshot.py --backend hf --model $D --no-remote-code --data-file $F $COMMON
  echo "===== $(date '+%H:%M') DeepSeek B  $DATA ====="
  CUDA_VISIBLE_DEVICES=$GPU_D $PY_D -u src/strategy_b.py --backend hf --model $D --no-remote-code --data-file $F --max-repairs $REPAIRS $COMMON
done
# C only on final_30: initial_10 was used to try out the agent (pilot), and all its tasks are in final_30
F=Dataset/mbpp_final_30.jsonl
echo "===== $(date '+%H:%M') Qwen C  final_30 ====="
CUDA_VISIBLE_DEVICES=$GPU $PY_Q -u src/strategy_c.py --backend hf --model $Q --data-file $F --max-steps $STEPS $COMMON
echo "===== $(date '+%H:%M') DeepSeek C  final_30 ====="
CUDA_VISIBLE_DEVICES=$GPU_D $PY_D -u src/strategy_c.py --backend hf --model $D --no-remote-code --data-file $F --max-steps $STEPS $COMMON
echo "===== $(date '+%H:%M') checks ====="
$PY_Q src/verify_extraction.py --quiet
$PY_Q src/reevaluate_sandbox.py
$PY_Q src/analyze_failures.py --data mbpp_final_30
$PY_Q src/analyze_failures.py --data mbpp_initial_10
echo "===== $(date '+%H:%M') ALL DONE ====="
