#!/bin/bash
# Step 6: Generate comparison plots
set -e
eval "$(conda shell.bash hook)"
conda activate vla
cd "$(dirname "$0")/../.."

echo "============================================="
echo "Step 6: Comparison Plots"
echo "============================================="
python -m vla.evaluate \
    --mode compare \
    --experiment_dirs \
        vla/checkpoints/sft_action \
        vla/checkpoints/sft_cot \
        vla/checkpoints/grpo_action \
        vla/checkpoints/grpo_cot \
    --labels \
        "SFT (action)" \
        "SFT (CoT)" \
        "GRPO (action)" \
        "GRPO (CoT)" \
    --output_path vla/comparison.png

echo "Done. Plot saved to vla/comparison.png"
