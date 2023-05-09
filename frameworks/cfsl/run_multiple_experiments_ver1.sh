#!/bin/bash
for arg in "$@"
do
  path=$arg
done
search_dir=$path
result_list_paths=""
for entry in "$search_dir"/*
do
  python train_continual_learning_few_shot_system.py --name_of_args_json_file $entry>v
  result_list_paths+=" "
  result_list_paths+=$(tail -n 1 v|rev|cut -d' ' -f 1|rev)
done
python calc_mean_std_from_csv.py $result_list_paths
