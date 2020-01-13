#!/bin/sh

export GPU_ID=$1

echo $GPU_ID

cd ..
export DATASET_DIR="/home/antreas/datasets/"
export CUDA_VISIBLE_DEVICES=$GPU_ID
# Activate the relevant virtual environment:
python train_continual_learning_few_shot_system.py --name_of_args_json_file experiment_config/mini-imagenet_cub_omniglot_cifar100_embedding_variant_standard_20_way_1_vgg-matching_network_shot__False_8_2_LSLR_conditioned_1.json --gpu_to_use $GPU_ID