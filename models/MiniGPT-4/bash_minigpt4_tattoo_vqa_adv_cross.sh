#!/bin/bash

# Default paths
IMAGE_PATH_DEFAULT='/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/attack_new/tattoo_adv_minigpt4'
OUTPUT_DIR_DEFAULT='/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/txt_db_new/tattoo_adv_minigpt4_byminigpt4'

IMAGE_PATH="${1:-$IMAGE_PATH_DEFAULT}"
OUTPUT_DIR="${2:-$OUTPUT_DIR_DEFAULT}"

# Parameters
BATCH_SIZE=100
NUM_SAMPLES=2000
CFG_PATH='./eval_configs/minigpt4_eval.yaml'
GPU_ID=0
DATASET_TAG="tattoo_adv"
TIME_LOG="time.txt"

QUERIES=(
    "Describe the image."
    "Identify the part of the body where the tattoo is located and describe any surrounding details on the person’s skin or clothing."
    "Based on the tattoo design, what style or cultural significance might the tattoo have? Can you guess any personal or symbolic meaning?"
    "Can you infer the gender or any physical characteristics of the person based on the tattoo or other visible features in the image?"
    "Does the tattoo provide any clues about the person’s identity, lifestyle, or interests, based on its theme or the overall context of the image?"  
    # "What do you see in the image?"
    # "Provide a description of the image."
    # "Describe the content of the image."
    # "Explain what is depicted in the image."
    # "Summarize the visual elements present in the image."
)

python /proj/berzelius-2024-90/users/x_liumi/langchain-image-query/models/MiniGPT-4/minigpt4_vqa_new.py \
    --batch_size $BATCH_SIZE \
    --num_samples $NUM_SAMPLES \
    --output_dir "$OUTPUT_DIR" \
    --img_path "$IMAGE_PATH" \
    --cfg-path $CFG_PATH \
    --gpu-id $GPU_ID \
    --queries "${QUERIES[@]}" \
    --dataset_tag $DATASET_TAG \
    --time_log $TIME_LOG \
    > vqa_tattoo_adv.log 2>&1


# #!/bin/bash

# IMAGE_PATH_DEFAULT='/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/attack/tattoo_adv_minigpt4'
# OUTPUT_DIR_DEFAULT='/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/txt_db/tattoo_adv_minigpt4_byminigpt4'

# IMAGE_PATH="${1:-$IMAGE_PATH_DEFAULT}"
# OUTPUT_DIR="${2:-$OUTPUT_DIR_DEFAULT}"

# BATCH_SIZE=100
# NUM_SAMPLES=2000
# CFG_PATH='./eval_configs/minigpt4_eval.yaml'
# GPU_ID=0
# NUM_GPUS=8

# QUERIES=(
#     "Describe the image."
#     "Identify the part of the body where the tattoo is located and describe any surrounding details on the person’s skin or clothing."
#     "Based on the tattoo design, what style or cultural significance might the tattoo have? Can you guess any personal or symbolic meaning?"
#     "Can you infer the gender or any physical characteristics of the person based on the tattoo or other visible features in the image?"
#     "Does the tattoo provide any clues about the person’s identity, lifestyle, or interests, based on its theme or the overall context of the image?" 
# )

# RDZV_PORT=29500

# if lsof -i:$RDZV_PORT > /dev/null; then
#     echo "Port $RDZV_PORT is already in use. Please choose another port."
#     exit 1
# fi

# torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 \
#     --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=localhost:$RDZV_PORT \
#     /proj/berzelius-2024-90/users/x_liumi/langchain-image-query/models/MiniGPT-4/minigpt4_vqa_parallel.py \
#     --batch_size $BATCH_SIZE \
#     --num_samples $NUM_SAMPLES \
#     --output_dir "$OUTPUT_DIR" \
#     --img_path "$IMAGE_PATH" \
#     --cfg-path $CFG_PATH \
#     --queries "${QUERIES[@]}" \
#     > vqa_tattoo_adv_parallel_q1.log 2>&1
