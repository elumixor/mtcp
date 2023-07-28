#!/bin/bash

echo "Running: $*"

# Run the command provided
eval "$*"

# Update the status
if [ $? -ne 0 ]; then
    echo "{ \"status\": \"failed\" }" > $MTCP_JOB_DIR/status.json
else
    echo "{ \"status\": \"done\" }" > $MTCP_JOB_DIR/status.json
fi

# Kill the tmux session
tmux kill-session -t $SESSION_ID