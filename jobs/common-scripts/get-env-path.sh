#!/bin/bash

# If env path is undefined, then point to the /eos/user/v/vyazykov/mtcp/venv
if [ -z "$ENV_PATH" ]; then
    # If the job dir is not definec, use the common env path
    if [ -z "$MTCP_JOB_DIR" ]; then
        export ENV_PATH="/eos/user/v/vyazykov/mtcp/venv"
    else
        # Otherwise, use the job dir
        export ENV_PATH="$MTCP_JOB_DIR/venv"
    fi
fi

echo "\$ENV_PATH = $ENV_PATH"