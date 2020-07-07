"""oneshot.py"""

import os
import json
import argparse

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets, transforms

from cls_module.cls import CLS

import utils

from omniglot_one_shot_dataset import OmniglotTransformation, OmniglotOneShotDataset
from lake_oneshot_metrics import LakeOneshotMetrics


def main():
  parser = argparse.ArgumentParser(description='Complementary Learning System: One-shot Learning Experiments')
  parser.add_argument('--config', nargs="?", type=str, default='config.json',
                      help='Configuration file for experiments.')

  args = parser.parse_args()

  with open(args.config) as config_file:
    config = json.load(config_file)

  utils.set_seed(config['seed'])

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  summary_dir = utils.get_summary_dir()

  writer = SummaryWriter(log_dir=summary_dir)

  image_tfms = transforms.Compose([
      transforms.ToTensor(),
      OmniglotTransformation(resize_factor=config['image_resize_factor'])
  ])

  image_shape = config['image_shape']
  pretrained_model_path = config['pretrained_model_path']

  # Pretraining
  # ---------------------------------------------------------------------------
  if not pretrained_model_path:
    background_loader = torch.utils.data.DataLoader(
        datasets.Omniglot('./data', background=True, download=True, transform=image_tfms),
        batch_size=config['pretrain_batch_size'], shuffle=True)

    model = CLS(image_shape, config, device=device, writer=writer).to(device)

    # Pre-train the model
    for epoch in range(1, config['pretrain_epochs'] + 1):
      model.train()

      for batch_idx, (data, target) in enumerate(background_loader):
        data, target = data.to(device), target.to(device)

        losses, _ = model(data, target)

        if batch_idx % 20 == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
              epoch, batch_idx * len(data), len(background_loader.dataset),
              100. * batch_idx / len(background_loader), losses['ltm'].item()))

    pretrained_model_path = os.path.join(summary_dir, 'pretrained_model_' + str(epoch) + '.pt')

    print('Saving model to:', pretrained_model_path)
    torch.save(model.state_dict(), pretrained_model_path)

  # Study and Recall
  # ---------------------------------------------------------------------------

  # Prepare data loaders
  study_loader = torch.utils.data.DataLoader(
      OmniglotOneShotDataset('./data', train=True, download=True,
                             transform=image_tfms, target_transform=None),
      batch_size=config['study_batch_size'], shuffle=False)

  recall_loader = torch.utils.data.DataLoader(
      OmniglotOneShotDataset('./data', train=False, download=True,
                             transform=image_tfms, target_transform=None),
      batch_size=config['study_batch_size'], shuffle=False)

  assert len(study_loader) == len(recall_loader)

  oneshot_dataset = enumerate(zip(study_loader, recall_loader))

  # Initialise metrics
  lake_oneshot_metrics = LakeOneshotMetrics()

  # Load the pretrained model
  model = CLS(image_shape, config, device=device, writer=writer).to(device)

  for idx, ((study_data, study_target), (recall_data, recall_target)) in oneshot_dataset:
    study_data, study_target = study_data.to(device), np.array(study_target)
    recall_data, recall_target = recall_data.to(device), np.array(recall_target)

    # Reset to saved model
    model.load_state_dict(torch.load(pretrained_model_path))

    # Study
    # --------------------------------------------------------------------------
    model.train()
    for _ in range(config['study_steps']):
      model(study_data, study_target, mode='study')

    # Recall
    # --------------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
      model(recall_data, recall_target, mode='recall')

      results = lake_oneshot_metrics.compute(model.features['study'], model.features['recall'], modes=['oneshot'])
      lake_oneshot_metrics.report(results)

      summary_names = ['study_inputs', 'recall_inputs', 'recall_stm']
      summary_images = []

      for name in summary_names:
        mode_key, feature_key = name.split('_')
        summary_features = model.features[mode_key][feature_key]
        summary_features = summary_features.permute(0, 2, 3, 1)

        summary_image = (name, summary_features, summary_features.data.shape)
        summary_images.append(summary_image)

      utils.add_completion_summary(summary_images, summary_dir, idx, save_figs=True)

  lake_oneshot_metrics.report_averages()

  writer.flush()
  writer.close()

if __name__ == '__main__':
  main()
