#!/bin/bash

# Default paths (overridden if passed in)
IMAGE_PATH_DEFAULT='/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/attack_new/car_adv_minigpt4'
OUTPUT_DIR_DEFAULT='/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/txt_db_new/car_adv_minigpt4_byminigpt4'

IMAGE_PATH="${1:-$IMAGE_PATH_DEFAULT}"
OUTPUT_DIR="${2:-$OUTPUT_DIR_DEFAULT}"

# Parameters
BATCH_SIZE=100
NUM_SAMPLES=2000
CFG_PATH='./eval_configs/minigpt4_eval.yaml'
GPU_ID=0
DATASET_TAG="car_adv"
TIME_LOG="time.txt"

QUERIES=(
    "Describe the image." 
    "Identify the car's make, model, and year." 
    "Describe the color and any distinguishing marks or features on the car." 
    "Read the license plate number and its country or region of registration." 
    "Describe the surroundings and location where the car is parked or moving."
    # "What do you see in the image?"
    # "Provide a description of the image."
    # "Describe the content of the image."
    # "Explain what is depicted in the image."
    # "Summarize the visual elements present in the image."
)

# Run the script
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
    > vqa_car_adv.log 2>&1


# #!/bin/bash

# # Default paths
# IMAGE_PATH_DEFAULT='/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/attack/car_adv_minigpt4'
# OUTPUT_DIR_DEFAULT='/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/txt_db/car_adv_minigpt4_byminigpt4'

# IMAGE_PATH="${1:-$IMAGE_PATH_DEFAULT}"
# OUTPUT_DIR="${2:-$OUTPUT_DIR_DEFAULT}"

# BATCH_SIZE=100
# NUM_SAMPLES=2000
# CFG_PATH='./eval_configs/minigpt4_eval.yaml'
# GPU_ID=0
# NUM_GPUS=8

# QUERIES=(
#     "Describe the image." 
#     "Identify the car's make, model, and year." 
#     "Describe the color and any distinguishing marks or features on the car." 
#     "Read the license plate number and its country or region of registration." 
#     "Describe the surroundings and location where the car is parked or moving."
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
#     > vqa_car_parallel_adv.log 2>&1
