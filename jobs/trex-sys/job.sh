#!/bin/bash

source /eos/user/v/vyazykov/TRExFitter/setup.sh

export MTCP_ROOT="/eos/user/v/vyazykov/mtcp"
cd $MTCP_ROOT/trex-fitter

config="configs/probs-sys.config"

trex-fitter n $config
trex-fitter w $config
trex-fitter d $config
trex-fitter f $config
trex-fitter r $config