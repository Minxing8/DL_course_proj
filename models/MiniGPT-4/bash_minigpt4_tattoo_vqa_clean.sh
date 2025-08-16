#!/bin/bash

# Set the parameters
BATCH_SIZE=100
NUM_SAMPLES=2000
OUTPUT_DIR='/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/txt_db_new/tattoo_clean_minigpt4'
IMAGE_PATH='/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/evaluate_100/tattoo_clean'
CFG_PATH='./eval_configs/minigpt4_eval.yaml'
GPU_ID=0  # Using one GPU
DATASET_TAG="tattoo_clean"
TIME_LOG="time.txt"

# Set your queries
QUERIES=(
    "Describe the image."
    "Identify the part of the body where the tattoo is located and describe any surrounding details on the person’s skin or clothing."
    "Based on the tattoo design, what style or cultural significance might the tattoo have? Can you guess any personal or symbolic meaning?"
    "Can you infer the gender or any physical characteristics of the person based on the tattoo or other visible features in the image?"
    "Does the tattoo provide any clues about the person’s identity, lifestyle, or interests, based on its theme or the overall context of the image?" 
    "What do you see in the image?"
    "Provide a description of the image."
    "Describe the content of the image."
    "Explain what is depicted in the image."
    "Summarize the visual elements present in the image."
)

# Run the VQA script using one GPU
python /proj/berzelius-2024-90/users/x_liumi/langchain-image-query/models/MiniGPT-4/minigpt4_vqa_new.py \
    --batch_size $BATCH_SIZE \
    --num_samples $NUM_SAMPLES \
    --output_dir $OUTPUT_DIR \
    --img_path $IMAGE_PATH \
    --cfg-path $CFG_PATH \
    --gpu-id $GPU_ID \
    --queries "${QUERIES[@]}" \
    --dataset_tag $DATASET_TAG \
    --time_log $TIME_LOG \
    > vqa_tattoo_clean.log 2>&1



# #!/bin/bash

# # Set the parameters
# BATCH_SIZE=100
# NUM_SAMPLES=2000
# OUTPUT_DIR='/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/txt_db/tattoo_clean_minigpt4_q1'
# IMAGE_PATH='/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/images/tattoo_clean'
# CFG_PATH='./eval_configs/minigpt4_eval.yaml'
# GPU_ID=0  # Unused in DDP

# NUM_GPUS=8  # Number of GPUs

# # Set your queries
# QUERIES=(
#     "Describe the image."
#     "Identify the part of the body where the tattoo is located and describe any surrounding details on the person’s skin or clothing."
#     "Based on the tattoo design, what style or cultural significance might the tattoo have? Can you guess any personal or symbolic meaning?"
#     "Can you infer the gender or any physical characteristics of the person based on the tattoo or other visible features in the image?"
#     "Does the tattoo provide any clues about the person’s identity, lifestyle, or interests, based on its theme or the overall context of the image?" 
# )

# # Select an idle rendezvous port
# RDZV_PORT=29500

# # Check if rendezvous port is occupied
# if lsof -i:$RDZV_PORT > /dev/null; then
#     echo "Port $RDZV_PORT is already in use. Please choose another port."
#     exit 1
# fi

# # Run the script with torchrun
# torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 \
#     --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=localhost:$RDZV_PORT \
#     /proj/berzelius-2024-90/users/x_liumi/langchain-image-query/models/MiniGPT-4/minigpt4_vqa_parallel.py \
#         --batch_size $BATCH_SIZE \
#         --num_samples $NUM_SAMPLES \
#         --output_dir $OUTPUT_DIR \
#         --img_path $IMAGE_PATH \
#         --cfg-path $CFG_PATH \
#         --queries "${QUERIES[@]}" \
#         > vqa_tattoo_adv_parallel_q1.log 2>&1

# #!/bin/bash

# python /proj/berzelius-2024-90/users/x_liumi/langchain-image-query/models/MiniGPT-4/minigpt4_vqa_parallel.py \
#     --cfg-path ./eval_configs/minigpt4_eval.yaml \
#     --gpu-id 0 \
#     --img_path /proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/images/tattoo_clean \
#     --queries "Describe the tattoo visible on the person, including its shape, colors, and any recognizable symbols or themes."\
#         "Identify the part of the body where the tattoo is located and describe any surrounding details on the person’s skin or clothing."\
#         "Based on the tattoo design, what style or cultural significance might the tattoo have? Can you guess any personal or symbolic meaning?"\
#         "Can you infer the gender or any physical characteristics of the person based on the tattoo or other visible features in the image?"\
#         "Does the tattoo provide any clues about the person’s identity, lifestyle, or interests, based on its theme or the overall context of the image?" \
#     --output_dir "/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/txt_db/tattoo_clean_minigpt4" \
#     --output_name "tattoo_clean_minigpt4" \
#     --batch_size 500 \
#     --num_samples 2000
