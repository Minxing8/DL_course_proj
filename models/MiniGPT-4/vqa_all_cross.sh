#!/bin/bash

# Base paths
IMG_BASE="/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/attack_new"
OUT_BASE="/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/txt_db_cross"

# Scripts
CAR_SCRIPT="bash_minigpt4_car_vqa_adv_cross.sh"
CELE_SCRIPT="bash_minigpt4_cele_vqa_adv_cross.sh"
TATTOO_SCRIPT="bash_minigpt4_tattoo_vqa_adv_cross.sh"

# Dataset variations
car_dirs=("car_adv_blip" "car_adv_blip2opt" "car_adv_blip2t5" "car_adv_llava" "car_adv_pali")
cele_dirs=("cele_adv_blip" "cele_adv_blip2opt" "cele_adv_blip2t5" "cele_adv_llava" "cele_adv_pali")
tattoo_dirs=("tattoo_adv_blip" "tattoo_adv_blip2opt" "tattoo_adv_blip2t5" "tattoo_adv_llava" "tattoo_adv_pali")

# Run 5 passes
for i in {0..4}; do
  echo "==== RUN $((i+1)) for CAR ===="
  bash $CAR_SCRIPT "$IMG_BASE/${car_dirs[$i]}" "$OUT_BASE/${car_dirs[$i]}_minigpt4"

  echo "==== RUN $((i+1)) for CELE ===="
  bash $CELE_SCRIPT "$IMG_BASE/${cele_dirs[$i]}" "$OUT_BASE/${cele_dirs[$i]}_minigpt4"

  echo "==== RUN $((i+1)) for TATTOO ===="
  bash $TATTOO_SCRIPT "$IMG_BASE/${tattoo_dirs[$i]}" "$OUT_BASE/${tattoo_dirs[$i]}_minigpt4"
done

echo "All MiniGPT-4 cross runs completed."
