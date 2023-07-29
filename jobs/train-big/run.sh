#!/bin/bash

# Clear the logs
rm -rf $MTCP_JOB_LOGS_DIR/*

source "$MTCP_JOBS_DIR/common-scripts/find-gpu.sh"

bash $MTCP_JOBS_DIR/common-scripts/tmux-run.sh $MTCP_ROOT/ml/train.sh $MTCP_JOB_DIR/train-config.yaml