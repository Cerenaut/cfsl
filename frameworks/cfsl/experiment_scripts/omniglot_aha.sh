#!/bin/sh

export GPU_ID=${1:-0}
export CONTINUE_FROM_EPOCH=${2:-latest}


echo $GPU_ID
echo $CONTINUE_FROM_EPOCH

export DATASET_DIR="datasets/"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Activate the relevant virtual environment:

python train_continual_learning_few_shot_system.py \
  --name_of_args_json_file experiment_config/omniglot_aha.json \
  --gpu_to_use $GPU_ID \
  --continue_from_epoch=$CONTINUE_FROM_EPOCH
