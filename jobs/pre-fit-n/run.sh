#!/bin/bash

# Source the trex fitter environment
source /eos/user/v/vyazykov/TRExFitter/setup.sh # And pray that it works...

# cd into the trex-fitter config directory
cd $MTCP_ROOT/trex-fitter

# Run the trex-fitter
trex-fitter n configs/prefit.config