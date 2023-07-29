#!/bin/bash

echo "Running: $*" | tee -a $MTCP_JOB_LOGS_DIR/log


# Run the command provided
((eval "$*" ; echo > $MTCP_JOB_DIR/.exit_code $?) | tee -a $MTCP_JOB_LOGS_DIR/out) 3>&1 1>&2 2>&3 | tee -a $MTCP_JOB_LOGS_DIR/err

# Read the exit code
read -r exit_code < $MTCP_JOB_DIR/.exit_code

rm $MTCP_JOB_DIR/.exit_code

# Update the status
if [ $exit_code -ne 0 ]; then
    echo "{ \"status\": \"failed\" }" > $MTCP_JOB_DIR/status.json
else
    echo "{ \"status\": \"done\" }" > $MTCP_JOB_DIR/status.json
fi

# Kill the tmux session
tmux kill-session -t $SESSION_ID