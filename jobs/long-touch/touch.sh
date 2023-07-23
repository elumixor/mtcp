#!/bin/bash

echo "before sleep"
echo "before speep (stderr)" >&2

# Wait for 1 second
sleep 1

echo "after sleep"
echo "after sleep (stderr)" >&2

touch $MTCP_JOB_ARTIFACTS_DIR/touched