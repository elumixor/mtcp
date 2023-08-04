#!/bin/bash

source $(dirname $(realpath $BASH_SOURCE))/get-env-path.sh

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source $ATLAS_LOCAL_ROOT_BASE/user/atlasLocalSetup.sh
lsetup "root 6.26.08-x86_64-centos7-gcc11-opt"
source $ENV_PATH/bin/activate
export PYTHONPATH=$VIRTUAL_ENV/lib64/python3.9/:$ROOTSYS/lib