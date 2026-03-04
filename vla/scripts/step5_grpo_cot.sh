#!/bin/bash
# Step 5: GRPO Training (text + action / CoT)
set -e
eval "$(conda shell.bash hook)"
conda activate vla
cd "$(dirname "$0")/../.."

echo "============================================="
echo "Step 5: GRPO Training (text + action / CoT)"
echo "============================================="
python -m vla.train_grpo_cot \
    --sft_checkpoint vla/checkpoints/sft_cot/best_model \
    --num_iterations 50 \
    --episodes_per_iter 10 \
    --samples_per_state 4 \
    --lr 1e-5 \
    --kl_coeff 0.1 \
    --eval_interval 5 \
    --eval_episodes 10 \
    --output_dir vla/checkpoints/grpo_cot

echo "Done. Checkpoint saved to vla/checkpoints/grpo_cot/"
