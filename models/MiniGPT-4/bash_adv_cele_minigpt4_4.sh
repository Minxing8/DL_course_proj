# #!/bin/bash

# # set the params
# BATCH_SIZE=32          
# NUM_SAMPLES=12000         
# STEPS=300              
# OUTPUT_DIR='/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/attack/cele_adv_minigpt4'
# IMAGE_PATH='/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/images/cele_clean'
# TARGET_PATH='/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/attack/targeted_image'
# CFG_PATH='./eval_configs/minigpt4_eval.yaml'
# GPU_ID=0               # unused in ddp

# NUM_GPUS=8             # number of GPUs

# # select an idle rendezvous port
# RDZV_PORT=29500

# # check if rendezvous port occupied
# if lsof -i:$RDZV_PORT > /dev/null; then
#     echo "Port $RDZV_PORT is already in use. Please choose another port."
#     exit 1
# fi

# # run adv with torchrun 
# torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=localhost:$RDZV_PORT \
#     /proj/berzelius-2024-90/users/x_liumi/langchain-image-query/models/MiniGPT-4/adv_img_minigpt4_parallel.py \
#         --batch_size $BATCH_SIZE \
#         --num_samples $NUM_SAMPLES \
#         --steps $STEPS \
#         --output $OUTPUT_DIR \
#         --image_path $IMAGE_PATH \
#         --target_path $TARGET_PATH \
#         --cfg-path $CFG_PATH \
#         > run_parallel_cele.log 2>&1



#!/bin/bash

# Set default parameters
BATCH_SIZE=32          # Adjusted to a manageable size
NUM_SAMPLES=12000
STEPS=200               # Align with adversarial script
OUTPUT_DIR='/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/attack/cele_adv_minigpt4_4'
IMAGE_PATH='/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/images/cele_clean_4'
TARGET_PATH='/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/attack/targeted_image'
CFG_PATH='./eval_configs/minigpt4_eval.yaml'
GPU_ID=0
CSV_PATH='/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/attack/cele_minigpt4_mapping_4.csv'

# Create output directory
mkdir -p $OUTPUT_DIR

# Execute the adversarial image generation script
python /proj/berzelius-2024-90/users/x_liumi/langchain-image-query/models/MiniGPT-4/adv_img_minigpt4.py \
    --batch_size $BATCH_SIZE \
    --num_samples $NUM_SAMPLES \
    --steps $STEPS \
    --output $OUTPUT_DIR \
    --image_path $IMAGE_PATH \
    --target_path $TARGET_PATH \
    --csv_path $CSV_PATH \
    --cfg-path $CFG_PATH \
    --gpu-id $GPU_ID
    # --options is optional and not needed unless overriding configurations
