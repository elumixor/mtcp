#!/bin/bash

# Activate the conda environment
eval "$(conda shell.bash hook)"
conda activate tth

# Get the current directory
DIR=$(dirname "$0")

# Get the config file
if [ -z "$1" ]; then
    CONFIG="$DIR/configs/small.yaml"
else
    # Check if the file exists
    if [ ! -f "$1" ]; then
        CONFIG="$DIR/configs/$1.yaml"
    else
        CONFIG="$1"
    fi

    if [ ! -f "$CONFIG" ]; then
        echo "Config file $CONFIG does not exist"
        exit 1
    fi
fi

# Run the training
python "$DIR/train.py" "$CONFIG"