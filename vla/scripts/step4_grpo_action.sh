#!/bin/bash
# Step 4: GRPO Training (direct action)
set -e
eval "$(conda shell.bash hook)"
conda activate vla
cd "$(dirname "$0")/../.."

echo "============================================="
echo "Step 4: GRPO Training (direct action)"
echo "============================================="
python -m vla.train_grpo \
    --sft_checkpoint vla/checkpoints/sft_action/best_model \
    --output_format action_only \
    --num_iterations 50 \
    --episodes_per_iter 10 \
    --samples_per_state 4 \
    --lr 1e-5 \
    --kl_coeff 0.1 \
    --eval_interval 5 \
    --eval_episodes 50 \
    --output_dir vla/checkpoints/grpo_action

echo "Done. Checkpoint saved to vla/checkpoints/grpo_action/"
