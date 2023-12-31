#!/bin/bash

set -e
set -x

my_path=$(dirname $(realpath $0))
cd $my_path

# find /eos/atlas/atlascerngroupdisk/phys-higgs/HSG8/multilepton_ttWttH/v08/v0801/systematics-full -name "*.root" | head -1 > files.txt
# find /eos/atlas/atlascerngroupdisk/phys-higgs/HSG8/multilepton_ttWttH/v08/v0801/systematics-full -name "*.root" > files.txt
find /eos/atlas/atlascerngroupdisk/phys-higgs/HSG8/multilepton_ttWttH/v08/v0801/systematics-full -name "*.root" | grep -v "NoNeed" > files.txt

small_dir="/eos/user/v/vyazykov/mtcp/friend_ntuples/output/small"
# rm -rf $small_dir
# mkdir -p $small_dir

export MTCP_JOB="produce-small"
afs_job_dir="/afs/cern.ch/user/v/vyazykov/.condor/$MTCP_JOB"

rm -rf $afs_job_dir
mkdir -p $afs_job_dir/logs/{out,err,log}

rm -rf logs
ln -s $afs_job_dir/logs

cp -t $afs_job_dir job.sh files.txt condor_submit.sub

cd $afs_job_dir

condor_submit condor_submit.sub