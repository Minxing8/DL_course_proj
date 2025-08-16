#!/bin/bash

# List of scripts to run
scripts=(
    # "bash_minigpt4_cele_vqa_clean.sh"
    # "bash_minigpt4_car_vqa_clean.sh"
    # "bash_minigpt4_tattoo_vqa_clean.sh"
    # "bash_minigpt4_cele_vqa_adv.sh"
    "bash_minigpt4_car_vqa_adv.sh"
    "bash_minigpt4_tattoo_vqa_adv.sh"
)

# Loop through each script and execute it
for script in "${scripts[@]}"; do
    echo "Running $script..."
    bash "$script"
    
    # Check if the script was executed successfully
    if [ $? -ne 0 ]; then
        echo "$script encountered an error. Exiting."
        exit 1
    fi
done

echo "All scripts ran successfully."
