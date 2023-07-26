#!/bin/bash

ENV_PATH=$MTCP_JOB_DIR/venv

# Exit if the virtual environment does not exist
if [ ! -d "$ENV_PATH" ]; then
    echo "Virtual environment not found!" >&2
    exit 1
fi

# Activate virtual environment
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source $ATLAS_LOCAL_ROOT_BASE/user/atlasLocalSetup.sh
lsetup "root 6.26.08-x86_64-centos7-gcc11-opt"
source $ENV_PATH/bin/activate

# Run the data processing
cd $MTCP_ROOT
python data_processing/main.py
