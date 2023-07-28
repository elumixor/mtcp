#!/bin/bash

# Check if venv already exists
VENV_PATH=$MTCP_ROOT/venv

if [ -d "$VENV_PATH" ]; then
    echo "venv already exists"
    source $VENV_PATH/bin/activate
    return 0
fi

# Use python if it exists, otherwise use python3
python_cmd=python
if ! command -v $python_cmd &> /dev/null
then
    if ! command -v python3 &> /dev/null
    then
        echo "python3 could not be found"
        exit 1
    fi

    python_cmd=python3
fi

echo "python_cmd=$python_cmd"

# Create venv
$python_cmd -m venv $VENV_PATH

# Activate venv
source $VENV_PATH/bin/activate

echo "ENV ACTIVATED"

# Install pip and required packages (paramiko)
# python -m pip install --upgrade pip
python -m pip install paramiko