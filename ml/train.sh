#!/bin/bash

ml Anaconda3

# Activate the conda environment
eval "$(conda shell.bash hook)"
conda activate tth

# Enable some cuda stuff
ml CUDA

# Run the training
python "$HOME/mtcp/ml/train.py" "$1"