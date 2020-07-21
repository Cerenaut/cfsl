"""VisualComponent class."""

import copy

import torch
import torch.optim as optim
import torch.nn.functional as F

from cls_module.memory.interface import MemoryInterface
from cls_module.components.sparse_autoencoder import SparseAutoencoder


class VisualComponent(MemoryInterface):
  """An implementation of a long-term memory module using sparse convolutional autoencoder."""

  global_key = 'ltm'
  local_key = 'vc'

  def build(self):
    """Build Visual Component as long-term memory module."""
    vc = SparseAutoencoder(self.input_shape, self.config).to(self.device)
    vc_optimizer = optim.AdamW(vc.parameters(), lr=self.config['learning_rate'])

    self.add_module(self.local_key, vc)
    self.add_optimizer(self.local_key, vc_optimizer)

    # Compute expected output shape
    with torch.no_grad():
      sample_output = vc.encode(torch.rand(1, *(self.input_shape[1:])))
      sample_output = self.prepare_encoding(sample_output)

      self.output_shape = list(sample_output.data.shape)
      self.output_shape[0] = -1

    if 'classifier' in self.config:
      self.build_classifier(input_shape=self.config['output_shape'])

  def forward_memory(self, inputs, targets, labels):
    """Perform an optimization step using the memory module."""
    del labels

    if self.vc.training:
      self.vc_optimizer.zero_grad()

    # Optionally use different stride at test time
    stride = self.config['stride']
    if not self.vc.training and 'eval_stride' in self.config:
      stride = self.config['eval_stride']

    encoding, decoding = self.vc(inputs, stride)
    loss = F.mse_loss(decoding, targets)

    if self.vc.training:
      loss.backward()
      self.vc_optimizer.step()

    output_encoding = self.prepare_encoding(encoding)

    outputs = {
        'encoding': encoding,
        'decoding': decoding,

        'output': output_encoding  # Designated output for linked modules
    }

    self.features = {
        'vc': output_encoding.detach(),
        'recon': decoding.detach()
    }

    return loss, outputs

  def prepare_encoding(self, encoding):
    """Postprocessing for the VC encoding."""
    encoding = encoding.detach()

    if self.config['output_pool_size'] > 1:
      pool_padding = (self.config['output_pool_size'] - 1) // 2

      encoding = F.max_pool2d(
          encoding,
          kernel_size=self.config['output_pool_size'],
          stride=self.config['output_pool_stride'],
          padding=pool_padding)

    if self.config['output_norm_per_sample']:
      frobenius_norm = torch.sqrt(
          torch.sum(torch.square(encoding),
                    dim=[1, 2, 3],
                    keepdim=True)
      )
      encoding = encoding / frobenius_norm

    return encoding
