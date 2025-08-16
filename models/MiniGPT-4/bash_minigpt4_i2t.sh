#!/bin/bash

python /proj/berzelius-2024-90/users/x_liumi/langchain-image-query/models/MiniGPT-4/minigpt4_i2t.py \
    --cfg-path ./eval_configs/minigpt4_eval.yaml \
    --gpu-id 0 \
    --img_path /proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/attack/adv_test_img_minigpt4 \
    --query "what is the content of this image?" \
    --output_path "/proj/berzelius-2024-90/users/x_liumi/langchain-image-query/data/txt_db/adv_test_minigpt4" \
    --batch_size 1 \
    --batch_size_in_gen 1 \
    --num_samples 10
