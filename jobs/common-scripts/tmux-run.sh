#!/bin/bash

# Start a new tmux session with a random id and detach from it
SESSION_ID=$(uuidgen)
tmux new-session -d -s $SESSION_ID

# Run the command in the tmux session
tmux send-keys -t $SESSION_ID "$MTCP_JOBS_DIR/common-scripts/in-tmux.sh $*" C-m

# Save the session id to a json file status.json
echo "{ \"status\": \"running\", \"tmux\": { \"session\": \"$SESSION_ID\" } }" > $MTCP_JOB_DIR/status.json