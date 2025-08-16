#!/bin/bash

IMAGE_BASE="/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/attack_new"
OUTPUT_BASE="/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/txt_db_cross"

SCRIPT="bash_minigpt4_car_vqa_adv_cross.sh"  # assumes revised script name

car_dirs=("car_adv_blip" "car_adv_blip2opt" "car_adv_blip2t5" "car_adv_llava" "car_adv_pali")

for i in {0..4}; do
  echo "==== [CAR] Pass $((i+1)) ===="
  bash $SCRIPT "$IMAGE_BASE/${car_dirs[$i]}" "$OUTPUT_BASE/${car_dirs[$i]}_minigpt4"
done

echo "✅ CAR dataset - All 5 passes completed."
