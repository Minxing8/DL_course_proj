#!/bin/bash

# Default paths
IMAGE_PATH_DEFAULT='/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/attack_new/cele_adv_minigpt4'
OUTPUT_DIR_DEFAULT='/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/txt_db_new/cele_adv_minigpt4_byminigpt4'

IMAGE_PATH="${1:-$IMAGE_PATH_DEFAULT}"
OUTPUT_DIR="${2:-$OUTPUT_DIR_DEFAULT}"

# Parameters
BATCH_SIZE=100
NUM_SAMPLES=2000
CFG_PATH='./eval_configs/minigpt4_eval.yaml'
GPU_ID=0
DATASET_TAG="cele_adv"
TIME_LOG="time.txt"

QUERIES=(
    "Describe the image."
    "Identify the person in the image and describe their outfit."
    "Identify the person in the image and state the location and surroundings where the person is."
    "Who is the person in the image and what activity is the person engaged in?"
    "Mention any context clues that indicate who the person is or where they are." 
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
    > vqa_cele_adv.log 2>&1


# #!/bin/bash

# IMAGE_PATH_DEFAULT='/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/attack/cele_adv_minigpt4_2'
# OUTPUT_DIR_DEFAULT='/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/txt_db/cele_adv_minigpt4_byminigpt4_2'

# IMAGE_PATH="${1:-$IMAGE_PATH_DEFAULT}"
# OUTPUT_DIR="${2:-$OUTPUT_DIR_DEFAULT}"

# BATCH_SIZE=100
# NUM_SAMPLES=12000
# CFG_PATH='./eval_configs/minigpt4_eval.yaml'
# GPU_ID=0
# NUM_GPUS=8

# QUERIES=(
#     "Describe the image."
#     "Identify the person in the image and describe their outfit."
#     "Identify the person in the image and state the location and surroundings where the person is."
#     "Who is the person in the image and what activity is the person engaged in?"
#     "Mention any context clues that indicate who the person is or where they are." 
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
#     > vqa_cele_parallel_adv2.log 2>&1
