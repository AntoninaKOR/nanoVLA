#!/bin/bash
# Quick training smoke test — runs a minimal version of the full pipeline
# to verify everything works end-to-end before long runs.
set -e

cd "$(dirname "$0")/../.."
echo "=== Quick Training Smoke Test ==="

# 1. SFT (action_only) — 10 steps
echo ""
echo ">>> Step 1/4: SFT action_only (10 steps)"
python -m vla.train_sft \
    --data_dir vla/trajectories \
    --output_format action_only \
    --output_dir vla/checkpoints/test_sft_action \
    --max_steps 10 \
    --batch_size 4 \
    --grad_accum_steps 1 \
    --log_interval 5 \
    --eval_interval 10 \
    --env_eval_interval 10 \
    --eval_episodes 5 \
    2>&1

# 2. SFT (cot) — 10 steps
echo ""
echo ">>> Step 2/4: SFT cot (10 steps)"
python -m vla.train_sft \
    --data_dir vla/trajectories \
    --output_format cot \
    --output_dir vla/checkpoints/test_sft_cot \
    --max_steps 10 \
    --batch_size 4 \
    --grad_accum_steps 1 \
    --log_interval 5 \
    --eval_interval 10 \
    --env_eval_interval 10 \
    --eval_episodes 5 \
    2>&1

# 3. GRPO (action) — 2 iterations
echo ""
echo ">>> Step 3/4: GRPO action (2 iterations)"
python -m vla.train_grpo \
    --sft_checkpoint vla/checkpoints/test_sft_action/final_model \
    --output_format action_only \
    --output_dir vla/checkpoints/test_grpo_action \
    --num_iterations 2 \
    --episodes_per_iter 2 \
    --samples_per_state 2 \
    --eval_interval 2 \
    --eval_episodes 5 \
    2>&1

# 4. GRPO (cot) — 2 iterations
echo ""
echo ">>> Step 4/4: GRPO cot (2 iterations)"
python -m vla.train_grpo_cot \
    --sft_checkpoint vla/checkpoints/test_sft_cot/final_model \
    --output_dir vla/checkpoints/test_grpo_cot \
    --num_iterations 2 \
    --episodes_per_iter 2 \
    --samples_per_state 2 \
    --eval_interval 2 \
    --eval_episodes 5 \
    2>&1

echo ""
echo "=== ALL SMOKE TESTS PASSED ==="
echo "Checkpoints saved in vla/checkpoints/test_*"
