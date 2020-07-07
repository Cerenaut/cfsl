"""Complementary Learning System module."""

import torch
import torch.nn as nn

from .memory import ltm, stm


class CLS(nn.Module):
  """Complementary Learning System module."""

  ltm_key = 'ltm'
  stm_key = 'stm'

  def __init__(self, input_shape, config, device=None, writer=None):
    super(CLS, self).__init__()

    self.config = config
    self.writer = writer
    self.device = device
    self.input_shape = input_shape

    self.step = 0

    if self.writer is None:
      self.writer = SummaryWriter()

    if self.device is None:
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.build()

    self.features = {}
    self.stm_modes = ['study', 'recall']

    for mode in self.stm_modes:
      self.features[mode] = {}

    self.initial_state = self.state_dict()

  def build(self):
    """Build and initialize the long-term and short-term memory modules."""
    ltm_config = self.config[self.ltm_key]
    ltm_ = ltm.VisualComponent(config=ltm_config,
                               input_shape=self.input_shape,
                               output_shape=None,
                               device=self.device,
                               writer=self.writer)

    stm_config = self.config[self.stm_key]
    stm_ = stm.FastNN(config=stm_config,
                      input_shape=ltm_.output_shape,
                      output_shape=self.input_shape,
                      device=self.device,
                      writer=self.writer)

    self.add_module(self.ltm_key, ltm_)
    self.add_module(self.stm_key, stm_)

  def freeze(self, names):
    depth = 2

    for name in names:
      name = name.split('.', maxsplit=depth)
      parent_module = getattr(self, name[0], None)

      if parent_module is None:
        continue

      if len(name) == 1:
        print('Freezing whole module =', name[0])
        parent_module.train(False)
        continue

      if len(name) > 1:
        for child_name, child_module in parent_module.named_modules():
          if child_name == name[1]:
            child_module.train(False)

  def load_state_dict(self, state_dict, strict=False):
    modified_state_dict = {}

    for state_key in state_dict:
      if state_key.startswith(self.ltm_key) and state_key in state_dict:
        modified_state_dict[state_key] = state_dict[state_key]
        continue

      modified_state_dict[state_key] = self.initial_state[state_key]

    super().load_state_dict(modified_state_dict, strict)

  def pretrain(self):
    pass

  def memorise(self):
    pass

  def recall(self):
    pass

  def consolidate(self):
    pass

  def forward(self, inputs, labels=None, mode='pretrain'):  # pylint: disable=arguments-differ
    """Perform an optimisation step with the entire CLS module."""
    losses = {}
    outputs = {}

    if mode in self.stm_modes:
      self.freeze([self.ltm_key + '.vc'])

    losses[self.ltm_key], outputs[self.ltm_key] = self.ltm(inputs=inputs, targets=inputs, labels=labels)

    if mode in self.stm_modes:
      stm_input = outputs[self.ltm_key].detach()  # Ensures no gradients pass through modules
      losses[self.stm_key], outputs[self.stm_key] = self.stm(inputs=stm_input, targets=inputs)

    if mode in self.stm_modes:
      self.features[mode]['inputs'] = inputs
      self.features[mode]['labels'] = labels
      self.features[mode][self.ltm_key] = outputs[self.ltm_key]
      self.features[mode][self.stm_key] = outputs[self.stm_key]

    self.step += 1

    return losses, outputs
