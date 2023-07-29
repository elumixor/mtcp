#!/bin/bash

# Exit if MTCP_REQUIRE_MEMORY is not set
if [ -z "$MTCP_REQUIRE_MEMORY" ]; then
    echo "MTCP_REQUIRE_MEMORY is not set" >&2 | tee $MTCP_JOB_LOGS_DIR/err
    exit 1
fi


# Require at least 20 GB
# Get the GPUs info
GPUs_info=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)

echo "Required memory: $MTCP_REQUIRE_MEMORY MB" | tee $MTCP_JOB_LOGS_DIR/log

echo "GPUs info:" | tee -a $MTCP_JOB_LOGS_DIR/log

i=0
found=false
max_mem=0
max_mem_index=0

while read GPU_free; do
    if [ $GPU_free -gt $MTCP_REQUIRE_MEMORY ]; then
        found=true
        echo "GPU $i has $GPU_free MB free memory (enough)" | tee -a $MTCP_JOB_LOGS_DIR/log
        if [ $GPU_free -gt $max_mem ]; then
            max_mem=$GPU_free
            max_mem_index=$i
        fi
    else
        echo "GPU $i has $GPU_free MB free memory (not enough)" | tee -a $MTCP_JOB_LOGS_DIR/log
    fi

    i=$((i+1))
done <<< "$GPUs_info"

if [ ! $found ]; then
    echo "No GPU has enough memory" >&2 | tee $MTCP_JOB_LOGS_DIR/err
    exit 1
fi

echo "Using GPU $max_mem_index with $max_mem MB free memory" | tee -a $MTCP_JOB_LOGS_DIR/log

export CUDA_VISIBLE_DEVICES=$max_mem_index
