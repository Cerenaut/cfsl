#!/bin/bash

echo
echo "##############   run-remote-screen.sh   ###############"
echo "#                                                     #"
echo "# NOTE: Will run from the script's containing folder  #"
echo "#                                                     #"
echo "#######################################################"

if [ "$1" == "-h" -o "$1" == "--help" ]; then
  echo "Usage: `basename $0` HOST KEY_FILE USER PORT"
  exit 0
fi

script=${1:-omniglot_aha.sh}
host=${2:-104.171.200.134}

echo "Run script = " $script
echo "Using host = " $host

ssh -p 22 ubuntu@${host} 'bash --login -s' <<ENDSSH

  cd \$HOME
  sudo rm -rf cerenaut-pt-core
  git clone https://github.com/Cerenaut/cerenaut-pt-core.git
  cd cerenaut-pt-core
  sudo python setup.py develop

  cd \$HOME
  sudo rm -rf pt-aha
  git clone https://github.com/Cerenaut/pt-aha.git
  cd pt-aha/cls_module
  sudo python setup.py develop

  export RUN_DIR=\$HOME/cfsl/frameworks/cfsl

  cd \$RUN_DIR
  sudo pip install -r requirements.txt

  export SCRIPT=\$RUN_DIR/experiment_scripts/$script

  # Start experiment in a new screen session, named "experiment"
  screen -dmS experiment bash -c '
    # Construct the run command
    cmd="sudo bash \$SCRIPT"
    echo \$cmd

    # On success, end screen session
    if eval \$cmd; then
      exit
    # Otherwise, maintain session to investigate errors
    else
      exec bash
    fi
  '

ENDSSH

status=$?

if [ $status -ne 0 ]
then
	echo "ERROR: Could not complete execute run-remote-screen.sh on remote machine through ssh." >&2
	echo "	Error status = $status" >&2
	echo "	Exiting now." >&2
	exit $status
fi
