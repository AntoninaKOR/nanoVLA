#!/bin/bash
# Step 1: Collect expert trajectories from MiniGrid EmptyEnv
set -e
eval "$(conda shell.bash hook)"
conda activate vla
cd "$(dirname "$0")/../.."

echo "============================================="
echo "Step 1: Collecting expert trajectories"
echo "============================================="
python -m vla.collect_trajectories \
    --num_episodes 5000 \
    --env_size 8 \
    --output_dir vla/trajectories \
    --seed 42

echo "Done. Trajectories saved to vla/trajectories/"
