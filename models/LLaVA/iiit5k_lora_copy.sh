#!/bin/bash

BATCH_SIZE=5
OUTPUT_DIR='/proj/berzelius-2024-90/users/x_liumi/DL_proj/data/txt_db/iiit5k_llava_loratrain200psudo_1000_ori_hybrid1to2_t1p1b2'
IMG_DIR='/proj/berzelius-2024-90/users/x_liumi/DL_proj/data/IIIT5K/evaluate_1000'

# Fine-tuned LoRA dir & base
MODEL_NAME='/proj/berzelius-2024-90/users/x_liumi/DL_proj/models/LLaVA/checkpoints/iiit5k-hybrid-lora-7b-train200psudo_1to2'
MODEL_BASE='liuhaotian/llava-v1.5-7b'

NUM_SAMPLES=10000
TEMPERATURE=0.1
TOP_P=1.0
NUM_BEAMS=2

QUESTIONS=(
  "What are the words in the image?"
)

python /proj/berzelius-2024-90/users/x_liumi/LLaVA/llava_vqa.py \
  --batch_size $BATCH_SIZE \
  --output_dir $OUTPUT_DIR \
  --image_dir $IMG_DIR \
  --model_name_or_path $MODEL_NAME \
  --model_base $MODEL_BASE \
  --num_samples $NUM_SAMPLES \
  --max_new_tokens 200 \
  --temperature $TEMPERATURE \
  --top_p $TOP_P \
  --num_beams $NUM_BEAMS \
  --questions "${QUESTIONS[@]}"
