"""MemoryInterface class."""

import torch
import torch.nn as nn

from cls_module.components.label_learner import LabelLearner

class MemoryInterface(nn.Module):
  """An implementation of a short-term memory module using a simple autoencoder."""

  global_key = 'memory'
  local_key = None

  def __init__(self, config, input_shape, target_shape=None, device=None, writer=None):
    super(MemoryInterface, self).__init__()

    self.config = config
    self.input_shape = input_shape
    self.target_shape = target_shape

    self.device = device
    self.writer = writer

    self.build()

    self.step = 0
    self.initial_state = self.state_dict()

    if self.global_key in self.config:
      self.config = self.config[self.global_key]

  def compute_output_shape(self, fn, input_shape=None):
    if input_shape is None:
      input_shape = self.input_shape

    with torch.no_grad():
      sample_output = fn(torch.rand(1, *(input_shape[1:])))
      output_shape = list(sample_output.data.shape)
      output_shape[0] = -1
    return output_shape

  def add_optimizer(self, name, optimizer):
    setattr(self, name + '_optimizer', optimizer)

  def build(self):
    raise NotImplementedError

  def build_classifier(self, input_shape):
    """Optionally build a classifier."""
    if not 'classifier' in self.config:
      raise KeyError('Classifier configuration not found.')

    classifier = LabelLearner(input_shape, self.config['classifier'])
    self.add_module('classifier', classifier)

  def forward_memory(self, inputs, targets, labels):
    raise NotImplementedError

  def forward(self, inputs, targets, labels=None):  # pylint: disable=arguments-differ
    """Perform an optimisation step with the entire CLS module."""
    losses = {}
    outputs = {}

    memory_loss, memory_outputs = self.forward_memory(inputs, targets, labels)

    losses['memory'] = {
        'loss': memory_loss
    }

    outputs['memory'] = memory_outputs
    outputs['classifier'] = {
        'predictions': None
    }

    if labels is not None:
      classifier_input = outputs['memory']['output'].detach()
      classifier_loss, preds = self.classifier(classifier_input, labels)

      losses['classifier'] = {
          'loss': classifier_loss
      }

      outputs['classifier']['predictions'] = preds

    self.step += 1

    return losses, outputs
