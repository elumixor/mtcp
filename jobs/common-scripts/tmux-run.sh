#!/bin/bash

# Start a new tmux session with a random id and detach from it
session_exists=true

# Generate a random session id
export SESSION_ID
while [ $session_exists ]; do
    SESSION_ID=$RANDOM
    tmux has-session -t $SESSION_ID 2>/dev/null
    session_exists=$([ $? -eq 0 ])

    if [ $session_exists ]; then
        echo "Session $SESSION_ID already exists"
    fi
done

tmux new-session -d -s $SESSION_ID

# Log the session id
echo "TMUX session started with id: $SESSION_ID" | tee -a $MTCP_JOB_LOGS_DIR/log

# Run the command in the tmux session
tmux send-keys -t $SESSION_ID "$MTCP_JOBS_DIR/common-scripts/in-tmux.sh $*" C-m

# Save the session id to a json file status.json
echo "{ \"status\": \"running\", \"tmux\": { \"session\": \"$SESSION_ID\" } }" > $MTCP_JOB_DIR/status.json