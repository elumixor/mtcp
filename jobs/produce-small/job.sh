#!/bin/bash

export MTCP_ROOT="/eos/user/v/vyazykov/mtcp"

source $MTCP_ROOT/jobs/common-scripts/get-env-path.sh

# cd to the MTCP root
cd $MTCP_ROOT

# Check if the environment is already activated
if [ -z "$VIRTUAL_ENV" ]; then
    # If not, activate it
    source $MTCP_ROOT/jobs/common-scripts/activate-python-env.sh
fi

# If the argument is provided, then use it as the file
if [ -z "$1" ]; then
    python friend_ntuples/produce_small.py
else
    python friend_ntuples/produce_small.py --file $1
fi