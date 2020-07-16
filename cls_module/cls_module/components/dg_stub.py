"""DGStub class."""

import torch
import torch.nn as nn


class DGStub(nn.Module):
  """A Dentate Gyrus (DG) stub module."""

  def __init__(self, input_shape, config):
    super(DGStub, self).__init__()

    self.config = config
    self.input_shape = input_shape

  def forward(self, x):  # pylint: disable=arguments-differ
    """Returns a batch of non overlapping n-hot samples in range [0,1]."""
    batch_size = x.shape[0]
    sample_size = self.config['num_units']
    n = self.config['sparsity']

    assert ((batch_size * n - 1) + n) < sample_size, "Can't produce batch_size {0} non-overlapping samples, " \
           "reduce n {1} or increase sample_size {2}".format(batch_size, n, sample_size)

    batch = torch.zeros(batch_size, sample_size)

    # return the sample at given idx
    for idx in range(batch_size):
      start_idx = idx * n
      end_idx = start_idx + n
      batch[idx][start_idx:end_idx] = 1

    return batch
