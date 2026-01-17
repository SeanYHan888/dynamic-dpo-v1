#!/bin/bash

# run_and_shutdown.sh
# Script to run a job and automatically shut down the RunPod instance.

MODE="rollout"
CONFIG_PATH="config_dpo.yaml"
NO_AUTO_SHUT=0

while [ $# -gt 0 ]; do
    case "$1" in
        rollout|sft|dpo)
            MODE="$1"
            ;;
        -noautoshut)
            NO_AUTO_SHUT=1
            ;;
        --config)
            CONFIG_PATH="$2"
            shift
            ;;
        *.yaml)
            CONFIG_PATH="$1"
            ;;
        *)
            echo "Unknown argument: $1 (use 'rollout', 'sft', 'dpo', '-noautoshut', or --config <path>)"
            exit 1
            ;;
    esac
    shift
done

if [ "$MODE" = "rollout" ]; then
    echo "Starting rollout data generation..."
    uv run python -m test.data_generation.run_rollout --config "$CONFIG_PATH"
    uv run python -m test.data_generation.upload_judged_rollout --config "$CONFIG_PATH"
    JOB_EXIT_CODE=$?
elif [ "$MODE" = "sft" ]; then
    echo "Starting SFT training..."
    # Note: The training script has an interactive prompt at the end for pushing to Hub.
    # If running unattented, we pipe 'yes' to auto-confirm if you WANT to push.
    # If you do NOT want to push, remove 'yes |' and configure push_to_hub: False in config.
    # Assuming user wants to run this unattended and likely wants to push if configured.
    yes | uv run train-sft --config "$CONFIG_PATH"
    JOB_EXIT_CODE=$?
elif [ "$MODE" = "dpo" ]; then
    echo "Starting DPO training..."
    uv run train-dpo --config "$CONFIG_PATH"
    JOB_EXIT_CODE=$?
else
    echo "Unknown mode: $MODE (use 'rollout', 'sft', or 'dpo')"
    exit 1
fi

if [ $JOB_EXIT_CODE -eq 0 ]; then
    echo "Job completed successfully."

    if [ "$MODE" = "rollout" ]; then
        OUTPUT_DIR=$(python - "$CONFIG_PATH" <<'PY'
import sys
try:
    import yaml
except Exception:
    print("rollout_output")
    sys.exit(0)
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
rollout = cfg.get("rollout", {}) if isinstance(cfg, dict) else {}
print(rollout.get("output_dir", "rollout_output"))
PY
)

        if [ -d "$OUTPUT_DIR" ]; then
            if [ -d "/mnt" ]; then
                DEST="/mnt/$(basename "$OUTPUT_DIR")"
                if [ -e "$DEST" ]; then
                    DEST="/mnt/$(basename "$OUTPUT_DIR")_$(date +%Y%m%d_%H%M%S)"
                fi
                echo "Moving dataset $OUTPUT_DIR -> $DEST"
                mv "$OUTPUT_DIR" "$DEST"
            else
                echo "/mnt not found; skipping dataset move."
            fi
        else
            echo "Output dir $OUTPUT_DIR not found; skipping dataset move."
        fi
    fi
else
    echo "Job failed with exit code $JOB_EXIT_CODE."
fi

# Shutdown logic
if [ $NO_AUTO_SHUT -eq 1 ]; then
    echo "Skipping auto-shutdown (flag -noautoshut set)."
elif [ -z "$RUNPOD_POD_ID" ]; then
    echo "RUNPOD_POD_ID is not set. Are you running in RunPod?"
    echo "Skipping auto-shutdown."
else
    echo "Shutting down pod $RUNPOD_POD_ID in 10 seconds..."
    sleep 10
    runpodctl stop pod "$RUNPOD_POD_ID"
fi
