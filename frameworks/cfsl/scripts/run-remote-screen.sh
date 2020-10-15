#!/bin/bash

if [ "$1" == "-h" -o "$1" == "--help" ]; then
  echo "Usage: `basename $0` HOST KEY_FILE USER PORT"
  exit 0
fi

host=${1:-incbox}
project=${2:-'cfsl/frameworks/cfsl'}  # relative to the AGI Code folder on incbox
anaenv=${3:-pagi_torch}
tensorboard=${4:-true}
viewfiles=${5:-true}

echo "Using host = " $host
echo "Run project = " $project
echo "With Anaconda env = " $anaenv

ssh "${host}" 'bash --login -s' <<ENDSSH

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

  # Start experiment in a new screen session, named "experiment"
  screen -dmS experiment bash -c '
    # Activate specified Anaconda environment
    source \$ACTIVATE $anaenv

    # Install Python dependencies
    pip install --quiet -r requirements.txt
    pip install -e \$RUN_DIR/cerenaut-pt-core

    # Construct the run command
    cmd="./omniglot_aha.sh 0 latest"
    echo \$cmd

    # On success, end screen session
    if eval \$cmd; then
      exit
    # Otherwise, maintain session to investigate errors
    else
      exec bash
    fi
  '

  # Tensorboard
  # ----------------------------------------------------
  if [ "$tensorboard" = true ]; then
    screen -X -S tensorboard quit
    screen -dmS tensorboard bash -c '
      source \$ACTIVATE $anaenv
      tensorboard --logdir=./runs
      exec bash
    '
  fi

  # View Files
  # ----------------------------------------------------
  if [ "$viewfiles" = true ]; then
    screen -X -S viewfiles quit
    screen -dmS viewfiles bash -c '
      cd ./runs
      python -m http.server
    '
  fi
ENDSSH

status=$?

if [ $status -ne 0 ]
then
	echo "ERROR: Could not complete execute run-remote-screen.sh on remote machine through ssh." >&2
	echo "	Error status = $status" >&2
	echo "	Exiting now." >&2
	exit $status
fi
