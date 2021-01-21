#!/bin/bash

echo
echo "################### remote-sync.sh ###################"
echo "## Sync a directory to ~/agief-remote-run           ##"
echo "## By default, the PWD                              ##"
echo "######################################################"

if [ "$1" == "-h" -o "$1" == "--help" ]; then
  echo "Usage: `basename $0` HOST KEY_FILE USER PORT"
  exit 0
fi

srcdir=${1:-$PWD}
host=${2:-incbox}

echo "Using srcdir= " $srcdir
echo "Using host = " $host

################################################################################
# sync code
################################################################################

DEST_DIR='~/agief-remote-run'

# sync this folder
cmd="rsync --chmod=ug=rwX,o=rX --perms -av $srcdir ${host}:$DEST_DIR --exclude='.git/' --filter=':- .gitignore'"
echo $cmd
eval $cmd
status=$?

if [ $status -ne 0 ]
then
  echo "ERROR:  Could not complete rsync operation - failed at 'sync this folder' stage." >&2
  echo "	Error status = $status" >&2
  echo "	Exiting now." >&2
  exit $status
fi
