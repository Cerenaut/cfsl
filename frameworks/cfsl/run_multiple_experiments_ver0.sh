#!/bin/bash
for arg in "$@"
do
  path=$arg
done
search_dir=$path
result_list_paths=()
for entry in "$search_dir"/*
do
  python train_continual_learning_few_shot_system.py --name_of_args_json_file $entry
done
for csv_path in "${result_list_paths[@]}"
do
  echo $csv_path
done
