#!/bin/bash

IMAGE_BASE="/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/attack_new"
OUTPUT_BASE="/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/txt_db_cross"

SCRIPT="bash_minigpt4_tattoo_vqa_adv_cross.sh"  # assumes revised script name

tattoo_dirs=("tattoo_adv_blip" "tattoo_adv_blip2opt" "tattoo_adv_blip2t5" "tattoo_adv_llava" "tattoo_adv_pali")

for i in {0..4}; do
  echo "==== [TATTOO] Pass $((i+1)) ===="
  bash $SCRIPT "$IMAGE_BASE/${tattoo_dirs[$i]}" "$OUTPUT_BASE/${tattoo_dirs[$i]}_minigpt4"
done

echo "✅ TATTOO dataset - All 5 passes completed."
