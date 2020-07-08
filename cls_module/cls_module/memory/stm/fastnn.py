"""FastNN class."""

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision

from cls_module.components.simple_autoencoder import SimpleAutoencoder


class FastNN(nn.Module):
  """An implementation of a short-term memory module using a simple autoencoder."""

  global_key = 'stm'
  local_key = 'fastnn'

  def __init__(self, config, input_shape, output_shape=None, device=None, writer=None):
    super(FastNN, self).__init__()

    self.config = config
    self.input_shape = input_shape
    self.output_shape = output_shape

    self.device = device
    self.writer = writer

    self.build()

    self.step = 0
    self.initial_state = self.state_dict()

    if self.global_key in self.config:
      self.config = self.config[self.global_key]

  def add_optimizer(self, name, optimizer):
    setattr(self, name + '_optimizer', optimizer)

  def summary_name(self, scope, name, mode=None):
    if mode is None:
      scope_module = getattr(self, scope)
      mode = 'train' if scope_module.training else 'eval'

    return self.global_key + '/' + scope + '/' + mode + '/' + name

  def build(self):
    """Build FastNN as short-term memory module."""
    fastnn = SimpleAutoencoder(self.input_shape, self.config, output_shape=self.output_shape).to(self.device)
    fastnn_optimizer = optim.AdamW(fastnn.parameters(),
                                   lr=self.config['learning_rate'],
                                   weight_decay=self.config['l2_penalty'])

    self.add_module(self.local_key, fastnn)
    self.add_optimizer(self.local_key, fastnn_optimizer)

  def forward_fastnn(self, inputs, targets):
    """Perform an optimization step using FastNN."""
    if self.fastnn.training:
      self.fastnn_optimizer.zero_grad()

    _, decoding = self.fastnn(inputs)

    loss = F.mse_loss(decoding, targets)

    if self.fastnn.training:
      loss.backward()
      self.fastnn_optimizer.step()

    if self.writer:
      self.writer.add_scalar(self.summary_name('fastnn', 'loss'), loss, self.step)
      self.writer.add_image(self.summary_name('fastnn', 'decoding'), torchvision.utils.make_grid(decoding), self.step)

    return loss, decoding

  def forward(self, inputs, targets):  # pylint: disable=arguments-differ
    """Perform an optimisation step with the entire CLS module."""
    loss, outputs = self.forward_fastnn(inputs, targets)

    self.step += 1

    return loss, outputs
