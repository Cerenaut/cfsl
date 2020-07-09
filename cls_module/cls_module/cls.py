"""Complementary Learning System module."""

import collections

import torch
import torch.nn as nn

import torchvision

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

    self.step = {}
    self.previous_mode = None

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
                               target_shape=None,
                               device=self.device,
                               writer=self.writer)

    stm_config = self.config[self.stm_key]
    stm_ = stm.FastNN(config=stm_config,
                      input_shape=ltm_.output_shape,
                      target_shape=self.input_shape,
                      device=self.device,
                      writer=self.writer)

    self.add_module(self.ltm_key, ltm_)
    self.add_module(self.stm_key, stm_)

  def reset(self):
    """Reset relevant sub-modules."""
    def weight_reset(m):
      if isinstance(m, (nn.Conv2d, nn.Linear)):
        m.reset_parameters()

    self.ltm.classifier.apply(weight_reset)
    self.ltm.classifier.optimizer.state = collections.defaultdict(dict)

    self.stm.apply(weight_reset)
    self.stm.classifier.optimizer.state = collections.defaultdict(dict)

  def freeze(self, names):
    """Selectively freeze sub-modules."""
    depth = 2

    for name in names:
      name = name.split('.', maxsplit=depth)
      parent_module = getattr(self, name[0], None)

      if parent_module is None:
        continue

      if len(name) == 1:
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

  def pretrain(self, inputs, labels):
    return self.forward(inputs, labels, mode='pretrain')

  def memorise(self, inputs, labels):
    return self.forward(inputs, labels, mode='study')

  def recall(self, inputs, labels):
    return self.forward(inputs, labels, mode='recall')

  def consolidate(self, inputs, labels):
    return self.forward(inputs, labels, mode='consolidate')

  def forward(self, inputs, labels=None, mode='pretrain'):  # pylint: disable=arguments-differ
    """Perform an optimisation step with the entire CLS module."""
    if mode not in self.step:
      self.step[mode] = 0

    losses = {}
    outputs = {}

    # Freeze ALL except LTM feature extractor for pretraining
    if mode == 'pretrain':
      self.train()
      self.freeze([self.ltm_key + '.classifier', self.stm_key])

    # Freeze LTM during memorisation
    elif mode == 'study':
      self.train()
      self.freeze([self.ltm_key])

    # Freeze ALL during recall
    elif mode == 'recall':
      self.eval()
      self.freeze([self.ltm_key, self.stm_key])

    # Freeze ALL except LTM classifier for consolidation
    elif mode == 'consolidate':
      self.train()
      self.freeze([self.ltm_key + '.vc', self.stm_key])

    # DEBUG: Check training status
    # if self.previous_mode != mode:
    #   print('Previous Mode =', self.previous_mode)
    #   print('Current Mode =', mode)

    #   self.previous_mode = mode

    #   for name, module in self.named_modules():
    #     if module.training:
    #       print(name, 'is training')

    losses[self.ltm_key], outputs[self.ltm_key] = self.ltm(inputs=inputs, targets=inputs, labels=labels)

    if mode in ['study', 'recall']:
      stm_input = outputs[self.ltm_key]['memory']['output'].detach()  # Ensures no gradients pass through modules

      losses[self.stm_key], outputs[self.stm_key] = self.stm(inputs=stm_input, targets=inputs, labels=labels)

      self.features[mode]['inputs'] = inputs
      self.features[mode]['labels'] = labels
      self.features[mode][self.ltm_key] = outputs[self.ltm_key]['memory']['encoding']
      self.features[mode][self.stm_key] = outputs[self.stm_key]['memory']['decoding']

    # Add summaries to TensorBoard
    if self.writer:
      summary_step = self.step[mode]

      for module_name in losses:
        for submodule_name in losses[module_name]:
          for metric_key, metric_value in losses[module_name][submodule_name].items():
            self.writer.add_scalar(mode + '/' + module_name + '/' + submodule_name + '/' + metric_key,
                                   metric_value,
                                   summary_step)

      self.writer.add_image(mode + '/inputs', torchvision.utils.make_grid(inputs), summary_step)

      for module_name in outputs:
        for submodule_name in outputs[module_name]:
          for metric_key, metric_value in outputs[module_name][submodule_name].items():
            if metric_key not in ['decoding']:
              continue

            self.writer.add_image(mode + '/' + module_name + '/' + submodule_name + '/' + metric_key,
                                  torchvision.utils.make_grid(metric_value),
                                  summary_step)

      self.writer.flush()

    self.step[mode] += 1

    return losses, outputs
