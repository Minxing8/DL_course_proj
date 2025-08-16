#!/bin/bash

BATCH_SIZE=5
OUTPUT_DIR='/proj/berzelius-2024-90/users/x_liumi/DL_proj/data/txt_db/iiit5k_llava_pretrained_500_t1p1b2'
IMG_DIR='/proj/berzelius-2024-90/users/x_liumi/DL_proj/data/IIIT5K/evaluate_500'
MODEL_NAME='liuhaotian/llava-v1.5-7b'
MODEL_BASE='None'
NUM_SAMPLES=10000
TEMPERATURE=0.1
TOP_P=1.0
NUM_BEAMS=2

QUESTIONS=(
    "What are the words in the image?" 
    # "Identify the car's make, model, and year." 
    # "Describe the color and any distinguishing marks or features on the car." 
    # "Read the license plate number and its country or region of registration." 
    # "Describe the surroundings and location where the car is parked or moving."
    # "What do you see in the image?"
    # "Provide a description of the image."
    # "Describe the content of the image."
    # "Explain what is depicted in the image."
    # "Summarize the visual elements present in the image."
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
