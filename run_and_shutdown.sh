#!/bin/bash

# run_and_shutdown.sh
# Script to run SFT training and automatically shut down the RunPod instance.

echo "Starting SFT training..."

# Run the training script.
# Note: The training script has an interactive prompt at the end for pushing to Hub.
# If running unattented, we pipe 'yes' to auto-confirm if you WANT to push.
# If you do NOT want to push, remove 'yes |' and configure push_to_hub: False in config.
# Assuming user wants to run this unattended and likely wants to push if configured.

# Check if push_to_hub is enabled in config to decide if we need 'yes'
# Ideally, we just run it. If prompt appears and no input, it hangs.
# So we use 'yes' to be safe for unattended run.
yes | python train_sft.py --config config_dpo.yaml

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully."
else
    echo "Training failed with exit code $TRAIN_EXIT_CODE."
fi

# Shutdown logic
if [ -z "$RUNPOD_POD_ID" ]; then
    echo "RUNPOD_POD_ID is not set. Are you running in RunPod?"
    echo "Skipping auto-shutdown."
else
    echo "Shutting down pod $RUNPOD_POD_ID in 10 seconds..."
    sleep 10
    runpodctl stop pod "$RUNPOD_POD_ID"
fi
