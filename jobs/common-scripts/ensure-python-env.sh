#!/bin/bash

source $(dirname $(realpath $0))/get-env-path.sh

# Check if the virtual environment exists
if [ -d "$ENV_PATH" ]; then
    echo "Virtual environment found!"
    exit 0
fi

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source $ATLAS_LOCAL_ROOT_BASE/user/atlasLocalSetup.sh
lsetup "root 6.26.08-x86_64-centos7-gcc11-opt"

# Create virtual environment. The --system-site-packages flag is needed to
# access ROOT libraries from the virtual environment
python3 -m venv $ENV_PATH --system-site-packages

# Activate virtual environment
source $ENV_PATH/bin/activate

# Reset the PYTHONPATH variable and add ROOT libraries to it
export PYTHONPATH=$VIRTUAL_ENV/lib64/python3.9/:$ROOTSYS/lib # not sure if it is needed

# Install required packages
python -m ensurepip --upgrade
python -m pip install --upgrade pip

# Install packages from requirements.txt
if [ ! -z "$MTCP_JOB_DIR" ]; then
    python -m pip install --upgrade -r $MTCP_JOB_DIR/requirements.txt
fi

# Deactivate virtual environment
echo "Environment created. Deactivating..."
deactivate
