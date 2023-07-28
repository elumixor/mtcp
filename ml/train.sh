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
    CONFIG="$DIR/configs/$1.yaml"
fi

# Run the training
python "$DIR/train.py" --config "$CONFIG"