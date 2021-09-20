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

host=${3:-104.171.200.134}
user=${5:-ubuntu}
port=${6:-22}

ssh -p $port ${user}@${host} 'bash --login -s' <<ENDSSH

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

  cd \$HOME/cfsl/frameworks/cfsl

  sudo pip install -r requirements.txt

  

ENDSSH

status=$?

if [ $status -ne 0 ]
then
	echo "ERROR: Could not complete execute run-remote-screen.sh on remote machine through ssh." >&2
	echo "	Error status = $status" >&2
	echo "	Exiting now." >&2
	exit $status
fi
