"""Classifier class."""

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class Classifier(nn.Module):
  """A simple, flexible and self-contained classifier module."""

  def __init__(self, input_shape, config):
    super(Classifier, self).__init__()

    self.config = config
    self.input_shape = list(input_shape)
    self.input_size = np.prod(self.input_shape[1:])

    self.hidden_units = self.config['hidden_units']
    self.output_units = self.config['output_units']

    if not isinstance(self.hidden_units, list):
      self.hidden_units = [self.hidden_units]

    layers = OrderedDict()
    in_features = self.input_size
    out_features = self.input_size

    for i, hidden_units in enumerate(self.hidden_units):
      out_features = hidden_units
      layers['layer_' + str(i)] = nn.Linear(in_features, out_features)
      layers['layer_' + str(i) + '_nonlinearity'] = nn.LeakyReLU()

    layers['layer_output'] = nn.Linear(out_features, self.output_units)

    self.model = nn.Sequential(layers)
    self.optimizer = optim.AdamW(self.model.parameters(),
                                 lr=self.config['learning_rate'],
                                 weight_decay=self.config['weight_decay'])
    self.loss = nn.CrossEntropyLoss()

  def forward(self, inputs, labels):  # pylint: disable=arguments-differ
    """Returns a batch of non overlapping n-hot samples in range [0,1]."""
    if self.training:
      self.optimizer.zero_grad()

    inputs = torch.flatten(inputs, start_dim=1)
    inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())

    preds = self.model(inputs)

    if not isinstance(labels, torch.Tensor):
      labels = torch.from_numpy(labels)

    loss = self.loss(preds, labels)

    if self.training:
      loss.backward()
      self.optimizer.step()

    return loss, preds
