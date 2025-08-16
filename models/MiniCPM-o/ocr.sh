#!/bin/bash
# run_minicpmo_ocr.sh

# --- user settings --------------------------------
BATCH_SIZE=16
OUTPUT_DIR="/home/labad/minxing/code/DL_proj/data/txt_db"
IMG_DIR="/home/labad/minxing/code/DL_proj/data/IIIT5K"
MODEL_NAME="openbmb/MiniCPM-o-2_6"
NUM_SAMPLES=10000
TEMPERATURE=0.0
TOP_P=1.0
NUM_BEAMS=1

QUESTIONS=(
    "What text is in the image?"
)
# --------------------------------------------------

python ocr.py \
  --model_name_or_path "$MODEL_NAME" \
  --image_dir "$IMG_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size $BATCH_SIZE \
  --num_samples $NUM_SAMPLES \
  --max_new_tokens 128 \
  --temperature $TEMPERATURE \
  --top_p $TOP_P \
  --num_beams $NUM_BEAMS \
  --questions "${QUESTIONS[@]}"
