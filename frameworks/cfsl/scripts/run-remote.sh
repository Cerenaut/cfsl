#!/bin/bash

if [ "$1" == "-h" -o "$1" == "--help" ]; then
  echo "Usage: `basename $0` HOST KEY_FILE USER PORT"
  exit 0
fi

host=${1:-incbox}
project=${2:-'cfsl/frameworks/cfsl'}  # relative to the AGI Code folder on incbox
anaenv=${3:-pagi_torch}

echo "Using host = " $host
echo "Run project = " $project
echo "With Anaconda env = " $anaenv

ssh ${host} 'bash --login -s' <<ENDSSH

  # Environment Setup
  # ----------------------------------------------------

  # Find the Anaconda environment on remote machnine
  export ACTIVATE=/media/data/anaconda3/bin/activate
  echo default activate file = \$ACTIVATE
  if [ ! -f \$ACTIVATE ]; then
    export ACTIVATE=~/anaconda3/bin/activate
  fi
  echo using activate file = \$ACTIVATE
  export RUN_DIR=\$HOME/agief-remote-run
  cd \$RUN_DIR/$project


  # Experiment
  # ----------------------------------------------------

  # Activate specified Anaconda environment
  source \$ACTIVATE $anaenv

  # Install dependencies
  pip install -e \$RUN_DIR/cerenaut-pt-core

  # Construct the run command
  cmd="./omniglot_aha.sh 0 latest"
  echo \$cmd
  eval \$cmd

ENDSSH

status=$?

if [ $status -ne 0 ]
then
	echo "ERROR: Could not complete execute run-remote.sh on remote machine through ssh." >&2
	echo "	Error status = $status" >&2
	echo "	Exiting now." >&2
	exit $status
fi
