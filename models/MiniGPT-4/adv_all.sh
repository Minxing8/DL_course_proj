#!/bin/bash

# List of scripts to run
scripts=(
    "bash_adv_tattoo_minigpt4.sh"
    "bash_adv_cele_minigpt4.sh"
    "bash_adv_car_minigpt4.sh"
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
