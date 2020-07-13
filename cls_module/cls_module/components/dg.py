"""dg.py"""

import torch
import torch.nn as nn

from cls_module.utils import get_top_k


class InhibitionMask():
  """Creates a mask of inhibited neurons based on fired neurons.

  Samples are multiplied by mask such that where mask = 1, a neuron is
          allowed to pass through, and where mask = 0, a neuron is
          inhibited. Each time the mask is updated, the inhibition is
          reduced by a factor of gamma.
  """

  def __init__(self, dim):
    super(InhibitionMask, self).__init__()

    # set firstdim to 1 so it works on a sample not a batch
    self.phi = torch.ones(1, dim)

  def update(self, top_k_sample, gamma=1):
    """Mask value is set to 0 at location of top_k

    Args:
        top_k_sample: (binary tensor) Sample of top_k values (as 0. or 1.)
        gamma: (float) range: 0-1. Amount to decrease inhibition each timestep.
    """

    # Check if value is binary, and not hopfield. This is required for the
    # decay funcyion to work properly
    for t in top_k_sample.unique():
      if t not in (0, 1):
        raise ValueError('Input must be binary (0., 1.) for inhibition mask')

    self.phi[self.phi < 1] += gamma
    self.phi[self.phi >= 1] = 1.0

    # Elementwise mul by (1 - top k)
    # Example: 1 - 1 = 0, making active top_k index = 0 in mask.
    self.phi = self.phi * (1 - top_k_sample)

  def reset(self):
    self.phi = torch.ones(10)

  def __call__(self):
    return self.phi

class DG(nn.Module):
  """
  Dentate Gyrus network

  Generates sparse output signals where neurons are inhibited directly after
  firing.
  """

  def __init__(self, input_size, output_size):
    super(DG, self).__init__()

    self.fc1 = nn.Linear(input_size, output_size)

    # Initialize uniform distribution
    nn.init.xavier_uniform_(self.fc1.weight)
    nn.init.zeros_(self.fc1.bias)

    # Initialize inhibition mask
    self.phi = InhibitionMask(D_out)

  def forward(self, x, k=10, mask_type="binary", gamma=0.01618):  # pylint: disable=arguments-differ
    """
    Args:
        x: (tensor) sizedtorch.Size([32, 225])
    """

    x = F.leaky_relu(self.fc1(x))

    # Apply inhibition to every sample
    for i in range(len(x[:,])):
      s = x[i, :]
      s = s.clone() * self.phi().clone()
      s = get_top_k(s, k, mask_type, -1, -1)
      self.phi.update(s, gamma)
      x[i, :] = s

    return x
