#!/bin/bash

export ENV_PATH=$MTCP_JOB_DIR/venv

# Exit if the virtual environment does not exist
if [ ! -d "$ENV_PATH" ]; then
    echo "Virtual environment not found!" >&2
    exit 1
fi

# Activate virtual environment
source $MTCP_JOBS_DIR/common-scripts/activate-python-env.sh

# Run the data processing
cd $MTCP_ROOT
python data_processing/main.py
