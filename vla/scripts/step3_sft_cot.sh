#!/bin/bash
# Step 3: SFT Training (CoT format)
set -e
eval "$(conda shell.bash hook)"
conda activate vla
cd "$(dirname "$0")/../.."

echo "============================================="
echo "Step 3: SFT Training (CoT format)"
echo "============================================="
python -m vla.train_sft \
    --hf_model lusxvr/nanoVLM-230M-8k \
    --data_dir vla/trajectories \
    --output_format cot \
    --batch_size 8 \
    --grad_accum_steps 2 \
    --max_steps 2000 \
    --lr_mp 1e-4 \
    --lr_backbone 5e-5 \
    --eval_interval 200 \
    --env_eval_interval 500 \
    --eval_episodes 10 \
    --output_dir vla/checkpoints/sft_cot

echo "Done. Checkpoint saved to vla/checkpoints/sft_cot/"
