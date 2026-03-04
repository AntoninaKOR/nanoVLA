#!/bin/bash
# =============================================================================
# Full VLA Pipeline: NanoVLM → NanoVLA for MiniGrid EmptyEnv
# =============================================================================
# Runs all steps sequentially. Each step can also be run independently:
#   bash vla/step1_collect.sh
#   bash vla/step2_sft_action.sh
#   bash vla/step3_sft_cot.sh
#   bash vla/step4_grpo_action.sh
#   bash vla/step5_grpo_cot.sh
#   bash vla/step6_compare.sh
# =============================================================================

set -e
cd "$(dirname "$0")/../.."

bash vla/scripts/step1_collect.sh
bash vla/scripts/step2_sft_action.sh
bash vla/scripts/step3_sft_cot.sh
bash vla/scripts/step4_grpo_action.sh
bash vla/scripts/step5_grpo_cot.sh
bash vla/scripts/step6_compare.sh

echo ""
echo "============================================="
echo "Pipeline complete!"
echo "============================================="
echo "Results:"
echo "  SFT (action):  vla/checkpoints/sft_action/"
echo "  SFT (CoT):     vla/checkpoints/sft_cot/"
echo "  GRPO (action): vla/checkpoints/grpo_action/"
echo "  GRPO (CoT):    vla/checkpoints/grpo_cot/"
echo "  Comparison:    vla/comparison.png"
echo "  SFT (action):  vla/checkpoints/sft_action/"
echo "  SFT (CoT):     vla/checkpoints/sft_cot/"
echo "  GRPO (action): vla/checkpoints/grpo_action/"
echo "  GRPO (CoT):    vla/checkpoints/grpo_cot/"
echo "  Comparison:    vla/comparison.png"
