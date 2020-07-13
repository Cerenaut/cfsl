#!/bin/sh

export GPU_ID=$1

echo $GPU_ID

export DATASET_DIR="datasets/"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Activate the relevant virtual environment:

python train_continual_learning_few_shot_system.py \
  --name_of_args_json_file omniglot_cls.json \
  --gpu_to_use $GPU_ID \
  --continue_from_epoch=latest
