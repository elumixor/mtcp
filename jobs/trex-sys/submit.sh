#!/bin/bash

my_path=$(dirname $(realpath $0))
cd $my_path

export MTCP_ROOT=/eos/user/v/vyazykov/mtcp
export MTCP_JOB="trex-sys"

afs_job_dir="/afs/cern.ch/user/v/vyazykov/.condor/$MTCP_JOB"
rm -rf $afs_job_dir
mkdir -p $afs_job_dir/logs/{out,err,log}
rm -rf logs
ln -s $afs_job_dir/logs
cp -t $afs_job_dir job.sh condor_submit.sub
cd $afs_job_dir

condor_submit condor_submit.sub