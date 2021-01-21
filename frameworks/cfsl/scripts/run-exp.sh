#!/bin/bash

###########################################################################################
#
# Specify experiments to choose with as few as two command line params, setup and experiment. 
# You can also be more specific.
# You will need to define $AGI_CODE_HOME
#
##########################################################################################
#
# usage example:       ../agi-tensorflow/scripts/run-exp.sh incbox def ./definitions/epw-mnist-pretrain.json screen true false
#

host=${1-"incbox"}     # Which machine:     incbox / walter / rodney / walterh / incboxh
script=${2:-"screen"}   # 'screen' = use run-remote-screen.sh / 'docker' = use run-remote-docker.sh / anything else = use run-remote.sh

# set and run the appropriate script
# --------------------------

if [ "$script" == "screen" ]; then
  run_script=run-remote-screen.sh
#  tensorboard=true  # launch TensorBoard
#  viewfiles=true    # launch python web server
#  exit_screen=true  # automatically terminate if completed successfully
else
  run_script=run-remote.sh
fi

"${BASH_SOURCE%/*}"/remote-sync.sh "$AGI_CODE_HOME"/cerenaut-pt-core $host
"${BASH_SOURCE%/*}"/remote-sync.sh "$AGI_CODE_HOME"/cfsl $host
"${BASH_SOURCE%/*}"/$run_script $host

