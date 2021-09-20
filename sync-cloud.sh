#!/bin/bash

################################################################################
# sync code
################################################################################

DEST_DIR='~/cfsl'
HOSTNAME='ubuntu@104.171.200.134'

# sync this folder
cmd="rsync --chmod=ug=rwX,o=rX --perms -av ./ $HOSTNAME:$DEST_DIR --exclude='.git/' --filter=':- .gitignore'"
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
