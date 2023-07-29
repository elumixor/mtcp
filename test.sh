#!/bin/bash

# set -x

export MTCP_ROOT=$HOME/mtcp
export MTCP_JOBS_DIR=$MTCP_ROOT/jobs
export MTCP_JOB=train-simple
export MTCP_JOB_DIR=$MTCP_JOBS_DIR/$MTCP_JOB
export MTCP_JOB_LOGS_DIR=$MTCP_JOB_DIR/logs

rm -rf $MTCP_JOB_LOGS_DIR/*

export MTCP_REQUIRE_MEMORY=10000 # MB

bash $MTCP_JOB_DIR/run.sh