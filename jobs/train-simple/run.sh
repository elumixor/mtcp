#!/bin/bash

# Get the GPUs info
GPUs_info=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)

# Iterate over each line
i=0
found=false
echo "$GPUs_info" | while read GPU_free; do
    # If the GPU memory free is greater than 1000 MB
    if [ $GPU_free -gt 1000 ]; then
        found=true
        echo "GPU $i has $GPU_free MB free memory"
        break
    fi
done

# If no GPU has enough memory, exit
if [ $found ]; then
    echo "No GPU has enough memory"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$i

bash $MTCP_JOBS_DIR/common-scripts/tmux-run.sh $MTCP_ROOT/ml/train.sh --config $MTCP_JOB_DIR/train-config.yaml