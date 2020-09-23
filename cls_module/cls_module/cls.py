"""Complementary Learning System module."""

import os
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

import numpy as np

import cls_module
from cls_module.utils import square_image_shape_from_1d
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
    ltm_type = self.config['ltm_type']

    if ltm_type == 'vc':
      ltm_class = ltm.VisualComponent
    elif ltm_type == 'vgg':
      ltm_class = ltm.VGG
    else:
      raise NotImplementedError('LTM type not supported: ' + ltm_type)

    ltm_ = ltm_class(config=ltm_config,
                     input_shape=self.input_shape,
                     target_shape=None,
                     device=self.device,
                     writer=self.writer)

    stm_type = self.config['stm_type']
    stm_config = self.config[self.stm_key]

    if stm_type == 'fastnn':
      stm_class = stm.FastNN
    elif stm_type == 'aha':
      stm_class = stm.AHA
    else:
      raise NotImplementedError('STM type not supported: ' + stm_type)

    stm_ = stm_class(config=stm_config,
                     input_shape=ltm_.output_shape,
                     target_shape=self.input_shape,
                     device=self.device,
                     writer=self.writer)

    self.add_module(self.ltm_key, ltm_)
    self.add_module(self.stm_key, stm_)

  def reset(self, names=None):
    """Reset relevant sub-modules."""
    if names is None:
      names = ['ltm.classifier', 'stm']

    if 'ltm.classifier' in names:
      self.ltm.classifier.reset()

    if 'stm' in names:
      self.stm.reset()

      if hasattr(self.stm, 'classifier'):
        self.stm.classifier.reset()

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

    # Load pre-trained VC weights from TensorFlow implementation
    load_tf_vc_weights = False

    if load_tf_vc_weights:
      module_dirpath = os.path.dirname(cls_module.__file__)
      weights_filepath = os.path.join(module_dirpath, '..', 'vc_weights.npz')

      vc_params = np.load(weights_filepath)
      vc_encoder_weight = torch.from_numpy(vc_params['weights']).permute(3, 2, 0, 1)
      vc_encoder_bias = torch.from_numpy(vc_params['encoding_bias'])

      self.ltm.vc.encoder.weight.data = vc_encoder_weight.to(self.device)
      self.ltm.vc.encoder.bias.data = vc_encoder_bias.to(self.device)

  def pretrain(self, inputs, labels):
    return self.forward(inputs, labels, mode='pretrain')

  def memorise(self, inputs, labels):
    return self.forward(inputs, labels, mode='study')

  def recall(self, inputs, labels):
    return self.forward(inputs, labels, mode='recall')

  def consolidate(self, inputs, labels):
    return self.forward(inputs, labels, mode='consolidate')

  def forward(self, inputs, labels=None, mode=None):  # pylint: disable=arguments-differ
    """Perform an optimisation step with the entire CLS module."""
    if labels is not None and not isinstance(labels, torch.Tensor):
      labels = torch.from_numpy(labels).to(self.device)

    if mode not in self.step:
      self.step[mode] = 0

    losses = {}
    outputs = {}
    accuracies = {}

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

    else:
      raise NotImplementedError('Mode not supported.')

    losses[self.ltm_key], outputs[self.ltm_key] = self.ltm(inputs=inputs, targets=inputs, labels=labels)
    preds = outputs[self.ltm_key]['classifier']['predictions']

    if preds is not None:
      softmax_preds = F.softmax(preds, dim=1).argmax(dim=1)
      accuracies[self.ltm_key] = torch.eq(softmax_preds, labels).data.cpu().float().mean()

    if mode in ['study', 'recall']:
      stm_input = outputs[self.ltm_key]['memory']['output'].detach()  # Ensures no gradients pass through modules

      losses[self.stm_key], outputs[self.stm_key] = self.stm(inputs=stm_input, targets=inputs, labels=labels)

      preds = outputs[self.stm_key]['classifier']['predictions']

      if preds is not None:
        softmax_preds = F.softmax(preds, dim=1).argmax(dim=1)
        accuracies[self.stm_key] = torch.eq(softmax_preds, labels).data.cpu().float().mean()

      self.features[mode]['inputs'] = inputs.detach().cpu()
      self.features[mode]['labels'] = labels.detach().cpu()

      for key, value in self.stm.features.items():
        self.features[mode][self.stm_key + '_' + key] = value

      for key, value in self.ltm.features.items():
        self.features[mode][self.ltm_key + '_' + key] = value

    # Add summaries to TensorBoard
    if self.writer:
      summary_step = self.step[mode]

      self.write_loss_summary(self.writer, losses, mode, summary_step)
      self.write_accuracy_summary(self.writer, accuracies, mode, summary_step)
      self.write_output_summaries(self.writer, outputs, mode, summary_step)

      self.writer.add_image(mode + '/inputs', torchvision.utils.make_grid(inputs), summary_step)

      self.writer.flush()

    self.step[mode] += 1

    return losses, outputs

  def write_loss_summary(self, writer, losses, mode, summary_step):
    for module_name in losses:
      for submodule_name in losses[module_name]:
        for metric_key, metric_value in losses[module_name][submodule_name].items():
          scope = mode + '/' + module_name + '/' + submodule_name + '/' + metric_key

          if isinstance(metric_value, dict):
            for submetric_key, submetric_value in metric_value.items():
              writer.add_scalar(scope + '/' + submetric_key, submetric_value, summary_step)
          else:
            writer.add_scalar(scope, metric_value, summary_step)

  def write_accuracy_summary(self, writer, accuracies, mode, summary_step):
    for module_name, accuracy_value in accuracies.items():
      writer.add_scalar(mode + '/' + module_name + '/classifier/accuracy', accuracy_value, summary_step)

  def write_output_summaries(self, writer, outputs, mode, summary_step):
    for module_name in outputs:
      for submodule_name in outputs[module_name]:
        for metric_key, metric_value in outputs[module_name][submodule_name].items():
          summary_image = None

          if metric_key == 'decoding':
            summary_image = torchvision.utils.make_grid(metric_value, normalize=True, scale_each=True)

          elif metric_key == 'encoding':
            square_image_shape, _ = square_image_shape_from_1d(np.prod(metric_value.shape[1:]))
            summary_image = torch.reshape(metric_value, [-1, 1, square_image_shape[1], square_image_shape[2]])
            summary_image = summary_image[0]

          if summary_image is not None:
            writer.add_image(mode + '/' + module_name + '/' + submodule_name + '/' + metric_key,
                             summary_image,
                             summary_step)
