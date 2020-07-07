"""VisualComponent class."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

import numpy as np

from cls_module.components.sparse_autoencoder import SparseAutoencoder


class VisualComponent(nn.Module):
  """An implementation of a long-term memory module using sparse convolutional autoencoder."""

  global_key = 'ltm'
  local_key = 'vc'

  def __init__(self, config, input_shape, output_shape=None, device=None, writer=None):
    super(VisualComponent, self).__init__()

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
    """Build Visual Component as long-term memory module."""
    vc = SparseAutoencoder(self.input_shape, self.config).to(self.device)
    vc_optimizer = optim.AdamW(vc.parameters(), lr=self.config['learning_rate'])

    self.add_module(self.local_key, vc)
    self.add_optimizer(self.local_key, vc_optimizer)

    # Compute expected output shape
    with torch.no_grad():
      sample_output = self.vc.encode(torch.rand(*(self.input_shape)))
      sample_output = self.prepare_output(sample_output)

      self.output_shape = list(sample_output.data.shape)
      self.output_shape[0] = -1

    classifier = nn.Sequential(
        # nn.Dropout(p=0.25),
        nn.Linear(np.prod(self.output_shape[1:]), self.config['classifier']['hidden_units']),
        # nn.Dropout(p=0.75),
        nn.Linear(self.config['classifier']['hidden_units'], self.config['classifier']['output_units'])
    )
    classifier_optimizer = optim.AdamW(classifier.parameters(),
                                       lr=self.config['classifier']['learning_rate'],
                                       weight_decay=self.config['classifier']['l2_penalty'],
                                       amsgrad=True)
    self.classifier_loss = nn.CrossEntropyLoss()

    self.add_module('classifier', classifier)
    self.add_optimizer('classifier', classifier_optimizer)

  def forward_vc(self, inputs, targets):
    """Perform an optimization step using the LTM-VC module."""
    if self.vc.training:
      self.vc_optimizer.zero_grad()

    encoding, decoding = self.vc(inputs)
    loss = F.mse_loss(decoding, targets)

    # Add summaries to TensorBoard
    self.writer.add_scalar(self.summary_name('vc', 'loss'), loss, self.step)
    self.writer.add_image(self.summary_name('vc', 'input'), torchvision.utils.make_grid(inputs), self.step)
    self.writer.add_image(self.summary_name('vc', 'decoding'), torchvision.utils.make_grid(decoding), self.step)

    if self.vc.training:
      loss.backward()
      self.vc_optimizer.step()

    return loss, encoding

  def forward_classifier(self, inputs, labels):
    if self.classifier.training:
      self.classifier_optimizer.zero_grad()

    inputs = torch.flatten(inputs, start_dim=1)
    outputs = self.classifier(inputs)

    labels = torch.from_numpy(labels)
    loss = self.classifier_loss(outputs, labels)

    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)

    # Add summaries to TensorBoard
    self.writer.add_scalar(self.summary_name('classifier', 'loss'), loss, self.step)
    self.writer.add_scalar(self.summary_name('classifier', 'accuracy'), accuracy, self.step)

    if self.classifier.training:
      loss.backward()
      self.classifier_optimizer.step()

    return loss, outputs

  def prepare_output(self, outputs):
    outputs = outputs.detach()

    if self.config['output_pool_size'] > 1:
      pool_padding = (self.config['output_pool_size'] - 1) // 2

      outputs = F.max_pool2d(
          outputs,
          kernel_size=self.config['output_pool_size'],
          stride=self.config['output_pool_stride'],
          padding=pool_padding)

    return outputs

  def forward(self, inputs, targets, labels=None):  # pylint: disable=arguments-differ
    """Perform an optimization step using the LTM module."""
    loss, outputs = self.forward_vc(inputs, targets)
    outputs = self.prepare_output(outputs)

    if labels is not None:
      _, self.predictions = self.forward_classifier(outputs, labels)

    self.step += 1

    return loss, outputs
