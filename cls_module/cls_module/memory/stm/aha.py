"""AHA class."""

import logging
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from cls_module.utils import build_topk_mask
from cls_module.memory.interface import MemoryInterface
from cls_module.components.dg import DG
from cls_module.components.simple_autoencoder import SimpleAutoencoder

def pc_to_unit(tensor):
  """
  From b[-1,1] to [0,1]
  This implementation only works assuming tensor is binary
  """
  return (tensor > 0).float()


def unit_to_pc_linear(tensor):
  """Input assumed to be unit range. Linearly scales to -1 <= x <= 1"""
  return (tensor * 2.0) - 1.0  # Theoretical range limits -1 : 1


def unit_to_pc_sparse(tensor):
  """ From b[0,1] to b[-1,1] or [0,1] to r[-1,1] """
  tensor[tensor == 0] = -1
  return tensor


def get_pc_topk_shift(tensor, sparsity):
  """Input tensor must be batch of vectors.
  Returns a vector per batch sample of the shift required to make Hopfield converge.
  Assumes knowledge of Hopfield fixed sparsity."""

  # Intuition: The output distribution must straddle the zero point to make hopfield work.
  # These are the values that should be positive.
  cue_top_k_mask = build_topk_mask(tensor, dim=1, k=int(sparsity + 1))

  y = tensor
  y_inv = 1.0 - y
  y_inv_masked = y_inv * cue_top_k_mask
  y_inv_masked_max, _ = torch.max(y_inv_masked, dim=1)  # max per batch sample
  y_masked_min = 1.0 - y_inv_masked_max

  cue_tanh_masked_min = (y_masked_min * 2.0) - 1.0  # scale these values
  shift = (0.0 - cue_tanh_masked_min).unsqueeze(1)

  return shift


def dg_to_pc(tensor):
  """ From sparse r[0,1] to b[-1,1]"""
  tensor = (tensor > 0).float()
  tensor[tensor == 0] = -1
  return tensor


class AHA(MemoryInterface):
  """An implementation of a short-term memory module using a AHA."""

  global_key = 'stm'
  local_key = 'aha'

  def reset(self):
    """Reset modules and optimizers."""
    for name, module in self.named_modules():
      # Reset the module parameters
      if hasattr(module, 'reset_parameters'):
        module.reset_parameters()

      # Reset the module optimizer
      optimizer_name = name + '_optimizer'
      if hasattr(self, optimizer_name):
        module_optimizer = getattr(self, optimizer_name)
        module_optimizer.state = defaultdict(dict)

    self.pc_buffer = []

  def build(self):
    """Build AHA as short-term memory module."""
    # Build the Pattern Separation (PS) module
    ps_output_shape = self.build_ps(ps_config=self.config['ps'], ps_input_shape=self.input_shape)

    # Build the Pattern Retrieval (PR) module
    self.build_pr(pr_config=self.config['pr'], pr_input_shape=self.input_shape, pr_target_shape=ps_output_shape)

    # Build the Pattern Completion (PC) module
    pc_output_shape = self.build_pc(pc_config=self.config['pc'], pc_input_shape=ps_output_shape)

    # Build the Pattern Mapping (PM) module
    self.build_pm(pm_config=self.config['pm'], pm_input_shape=pc_output_shape, pm_target_shape=self.target_shape)

    # Build the Label Learner module
    if 'classifier' in self.config:
      self.build_classifier(input_shape=pc_output_shape)

    self.output_shape = pc_output_shape

  def build_ps(self, ps_config, ps_input_shape):
    """Build the Pattern Separation (PS) module."""
    ps = DG(ps_input_shape, ps_config)
    ps_output_shape = [1, ps_config['num_units']]

    self.add_module('ps', ps)

    return ps_output_shape

  def forward_ps(self, inputs):
    return self.ps(inputs)

  def build_pr(self, pr_config, pr_input_shape, pr_target_shape):
    """Builds the Pattern Retrieval (PR) module."""
    pr = SimpleAutoencoder(pr_input_shape, pr_config, output_shape=pr_target_shape)
    pr_optimizer = optim.AdamW(pr.parameters(),
                               lr=pr_config['learning_rate'],
                               weight_decay=pr_config['weight_decay'])

    self.add_module('pr', pr)
    self.add_optimizer('pr', pr_optimizer)

  def forward_pr(self, inputs, targets):
    """Perform one step using the PR module to learn to replicate the patterns generated using the PS."""
    pr_config = self.config['pr']

    if self.pr.training:
      self.pr_optimizer.zero_grad()

    _, logits = self.pr(inputs)
    loss = F.binary_cross_entropy_with_logits(logits, targets)

    if self.pr.training:
      loss.backward()
      self.pr_optimizer.step()

    y = torch.sigmoid(logits)
    y = y.detach()

    # Clip
    y = torch.clamp(y, 0.0, 1.0)

    # Normalize to [0, 1]
    # y = (y - y.min()) / (y.max() - y.min())

    # Sparsen
    if pr_config['sparsen']:
      k_pr = int(pr_config['sparsity'] * pr_config['sparsity_boost'])
      logging.info('PR Sparsen enabled k = %s', str(k_pr))
      mask = build_topk_mask(y, dim=1, k=k_pr)
      y = y * mask

    # Sum norm (all input is positive)
    # We expect a lot of zeros, or near zeros, and a few larger values.
    if pr_config['sum_norm'] > 0.0:
      logging.info('PR Sum-norm enabled')
      eps = 1e-13
      y_sum = torch.sum(y, dim=1, keepdim=True)
      reciprocal = 1.0 / y_sum + eps
      y = y * reciprocal * pr_config['sum_norm']

    # Softmax norm
    if pr_config['softmax']:
      logging.info('PR Softmax enabled')
      y = F.softmax(y)

    # After norm
    if pr_config['gain'] != 1.0:
      logging.info('PR Gain enabled')
      y = y * pr_config['gain']

    pr_out = y  # Unit range
    z_cue_in = pr_out

    # Range shift from unit to signed unit
    if pr_config['shift_range']:
      z_cue_in = unit_to_pc_linear(pr_out)  # Theoretical range limits [-1, 1]

    z_cue_shift = z_cue_in

    # Shift until k bits are > 0, i.e. min *masked* value should become equal to zero.
    if pr_config['shift_bits']:
      logging.info('PR Shift enabled')
      shift = get_pc_topk_shift(pr_out, pr_config['sparsity'])
      z_cue_shift = z_cue_in + shift

    outputs = {
        'pr_out': pr_out.detach(),
        'z_cue_in': z_cue_in.detach(),
        'z_cue': z_cue_shift.detach()
    }

    return loss, outputs['z_cue']

  def build_pc(self, pc_config, pc_input_shape):
    """Builds the Pattern Completion (PC) module."""
    del pc_config

    # Initialise the buffer
    self.pc_buffer = []

    return pc_input_shape

  def forward_pc(self, inputs):
    """
    During training, store the inputs from PS into the buffer. At test time, use the inputs from the PR to lookup the
    matching PS pattern from the buffer using K-nearest neighbour with K=1.
    """
    pc_config = self.config['pc']

    if self.training:
      # Range shift from unit to signed unit
      if pc_config['shift_range']:
        inputs = unit_to_pc_linear(inputs)

      # Memorise inputs in buffer
      self.pc_buffer = inputs

      return self.pc_buffer

    recalled = torch.zeros_like(inputs)

    for i, test_input in enumerate(inputs):
      dist = torch.norm(self.pc_buffer - test_input, dim=1, p=None)
      knn = dist.topk(k=1, largest=False)
      recalled[i] = self.pc_buffer[knn.indices]

    return recalled

  def build_pm(self, pm_config, pm_input_shape, pm_target_shape):
    pm = SimpleAutoencoder(pm_input_shape, pm_config, output_shape=pm_target_shape)
    pm_optimizer = optim.AdamW(pm.parameters(),
                               lr=pm_config['learning_rate'],
                               weight_decay=pm_config['weight_decay'])

    self.add_module('pm', pm)
    self.add_optimizer('pm', pm_optimizer)

  def forward_pm(self, inputs, targets):
    """
    Perform one step using the PM module to learn to reconstruct input image using the patterns from the PC.
    """
    if self.pm.training:
      self.pm_optimizer.zero_grad()

    encoding, decoding = self.pm(inputs)

    loss = F.mse_loss(decoding, targets)

    outputs = {
        'encoding': encoding.detach(),
        'decoding': decoding.detach(),

        'output': encoding.detach()  # Designated output for linked modules
    }

    if self.pm.training:
      loss.backward()
      self.pm_optimizer.step()

    return loss, outputs

  def forward_memory(self, inputs, targets, labels):
    """Perform one step using the entire system (i.e. all sub-modules of AHA)."""
    del labels

    losses = {}
    outputs = {}

    outputs['ps'] = self.forward_ps(inputs)

    losses['pr'], outputs['pr'] = self.forward_pr(inputs=inputs, targets=outputs['ps'])

    pc_input = outputs['ps'] if self.training else outputs['pr']
    outputs['pc'] = self.forward_pc(inputs=pc_input)

    losses['pm'], outputs['pm'] = self.forward_pm(inputs=outputs['pc'], targets=targets)

    outputs['encoding'] = outputs['pc'].detach()
    outputs['decoding'] = outputs['pm']['decoding'].detach()
    outputs['output'] = outputs['pc'].detach()

    self.features = {
        'ps': outputs['ps'].detach(),
        'pr': outputs['pr'].detach(),
        'pc': outputs['pc'].detach(),

        'recon': outputs['pm']['decoding'].detach()
    }

    return losses, outputs
