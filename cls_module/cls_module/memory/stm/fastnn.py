"""FastNN class."""

import torch
import torch.optim as optim
import torch.nn.functional as F

from cls_module.components.simple_autoencoder import SimpleAutoencoder
from cls_module.memory.interface import MemoryInterface


class FastNN(MemoryInterface):
  """An implementation of a short-term memory module using a simple autoencoder."""

  global_key = 'stm'
  local_key = 'fastnn'

  def build(self):
    """Build FastNN as short-term memory module."""
    fastnn = SimpleAutoencoder(self.input_shape, self.config, output_shape=self.target_shape).to(self.device)
    fastnn_optimizer = optim.AdamW(fastnn.parameters(),
                                   lr=self.config['learning_rate'],
                                   weight_decay=self.config['l2_penalty'])

    self.add_module(self.local_key, fastnn)
    self.add_optimizer(self.local_key, fastnn_optimizer)

    # Compute expected output shape
    with torch.no_grad():
      sample_output = fastnn.encode(torch.rand(1, *(self.input_shape[1:])))
      self.output_shape = list(sample_output.data.shape)
      self.output_shape[0] = -1

    if 'classifier' in self.config:
      self.build_classifier(input_shape=self.output_shape)

  def forward_memory(self, inputs, targets, labels):
    """Perform an optimization step using memory module."""
    del labels

    if self.fastnn.training:
      self.fastnn_optimizer.zero_grad()

    encoding, decoding = self.fastnn(inputs)

    loss = F.mse_loss(decoding, targets)

    outputs = {
        'encoding': encoding,
        'decoding': decoding,

        'output': encoding.detach()  # Designated output for linked modules
    }

    if self.fastnn.training:
      loss.backward()
      self.fastnn_optimizer.step()

    return loss, outputs
