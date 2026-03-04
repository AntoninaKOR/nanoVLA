#!/bin/bash
# Step 2: SFT Training (action_only format)
set -e
eval "$(conda shell.bash hook)"
conda activate vla
cd "$(dirname "$0")/../.."

echo "============================================="
echo "Step 2: SFT Training (action_only format)"
echo "============================================="
python -m vla.train_sft \
    --hf_model lusxvr/nanoVLM-230M-8k \
    --data_dir vla/trajectories \
    --output_format action_only \
    --batch_size 8 \
    --grad_accum_steps 2 \
    --max_steps 2000 \
    --lr_mp 1e-4 \
    --lr_backbone 5e-5 \
    --eval_interval 200 \
    --env_eval_interval 500 \
    --eval_episodes 50 \
    --output_dir vla/checkpoints/sft_action

echo "Done. Checkpoint saved to vla/checkpoints/sft_action/"
