#!/bin/bash

IMAGE_BASE="/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/attack_new"
OUTPUT_BASE="/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/txt_db_cross"

SCRIPT="bash_minigpt4_cele_vqa_adv_cross.sh"  # assumes revised script name

cele_dirs=("cele_adv_blip" "cele_adv_blip2opt" "cele_adv_blip2t5" "cele_adv_llava" "cele_adv_pali")

for i in {0..4}; do
  echo "==== [CELE] Pass $((i+1)) ===="
  bash $SCRIPT "$IMAGE_BASE/${cele_dirs[$i]}" "$OUTPUT_BASE/${cele_dirs[$i]}_minigpt4"
done

echo "✅ CELE dataset - All 5 passes completed."
