import torch
import torch.nn as nn
import torch.nn.functional as F

class DGStub(nn.Module):
  def __init__(self, config):
    super(DGStub, self).__init__()

    self.batch_size = self.config['batch_size']
    self.filters = self.config['filters']
    self.sparsity = self.config['sparsity']


  def forward(self, x):  # pylint: disable=arguments-differ
    """Returns a batch of non overlapping n-hot samples in range [0,1]."""
    del x

    batch_size = self.batch_size
    sample_size = self.filters
    n = self.sparsity

    assert ((batch_size * n - 1) + n) < sample_size, "Can't produce batch_size {0} non-overlapping samples, " \
           "reduce n {1} or increase sample_size {2}".format(batch_size, n, sample_size)

    batch = np.zeros(shape=(batch_size, sample_size))

    # return the sample at given idx
    for idx in range(batch_size):
      start_idx = idx * n
      end_idx = start_idx + n
      batch[idx][start_idx:end_idx] = 1

    return batch
