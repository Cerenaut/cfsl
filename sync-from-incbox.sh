#!/bin/bash

################################################################################
# sync code
################################################################################

REMOTE_DIR='~/agief-remote-run/cfsl/frameworks/cfsl/runs/omniglot_aha/saved_models'
LOCAL_DIR='~/Dev/cfsl/frameworks/cfsl/runs/omniglot_aha/saved_models'

# sync this folder
cmd="rsync --chmod=ug=rwX,o=rX --perms -av incbox:$REMOTE_DIR $LOCAL_DIR --exclude='.git/' " # --filter=':- .gitignore'"
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
