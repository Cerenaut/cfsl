"""Parse CFSL training loss from Jenkins output."""

import os
import re
import argparse

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input_path', type=str, help='an integer for the accumulator')

args = parser.parse_args()

if args.input_path is None or args.input_path == '' or not os.path.exists(args.input_path):
  raise ValueError('Input path does not exist.')

input_path = args.input_path
output_path = input_path[:-4] + '.csv'


with open(input_path, 'r') as f:
  data = f.readlines()

key = 'train_loss_mean'

headings = []
metrics = {}

for num, line in enumerate(data):
  if key in line:
    # Format the line
    line = re.sub(r'^.*?' + key, key, line).strip()

    # Extract metrics by splitting
    line = [x.strip() for x in line.split(',')]

    for metric in line:
      if metric == '':
        continue
      print(metric)

      split_metric = metric.split(':')

      if len(split_metric) > 2:
        continue

      metric_key, metric_value = (x.strip() for x in split_metric)

      if metric_key not in metrics:
        metrics[metric_key] = []

      metrics[metric_key].append(float(metric_value))

for label in metrics:
  plt.title(label)
  plt.plot(metrics[label])
  plt.xlabel('Epochs')
  plt.savefig(label + '.png')
  plt.close()
