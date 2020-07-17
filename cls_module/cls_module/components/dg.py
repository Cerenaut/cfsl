"""DentateGyrus class."""

import torch
import torch.nn as nn

import numpy as np

import cls_module.utils as utils


class DG(nn.Module):
  """
  A non-trainable module based on Dentate Gyrus (DG), produces sparse outputs and inhibits neurons after firing.
  """

  def __init__(self, input_shape, config):
    super(DG, self).__init__()

    self.config = config

    self.input_shape = list(input_shape)
    self.input_size = np.prod(self.input_shape[1:])

    self.layer = nn.Linear(self.input_size, self.config['num_units'], bias=False)
    self.layer.weight.requires_grad = False

    self.initialize()
    # nn.init.xavier_uniform_(self.layer.weight)

  def initialize(self):
    """Custom initialization *does* make a big difference to orthogonality, even with inhibition"""
    input_area = self.input_size
    hidden_size = self.config['num_units']

    num_weights = hidden_size * input_area
    random_values = torch.rand(num_weights)

    knockout_rate = self.config['knockout_rate']
    keep_rate = 1.0 - knockout_rate
    initial_mask = np.random.choice([0, 1], size=(num_weights), p=[knockout_rate, keep_rate])
    initial_values = random_values * initial_mask * self.config['init_scale']
    initial_values = initial_values.float()

    for i in range(0, hidden_size):
      w_sum = 0.0

      for j in range(0, input_area):
        offset = j * hidden_size + i
        w_ij = initial_values[offset]
        w_sum = w_sum + abs(w_ij)

      w_norm = 1.0 / w_sum

      for j in range(0, input_area):
        offset = j * hidden_size + i
        w_ij = initial_values[offset]
        w_ij = w_ij * w_norm
        initial_values[offset] = w_ij

    self.layer.weight.data = torch.reshape(initial_values, shape=(hidden_size, input_area))

  def apply_sparse_filter(self, encoding):
    """Sparse filtering with inhibition."""
    hidden_size = self.config['num_units']
    batch_size = encoding.shape[0]

    k = int(self.config['sparsity'])
    inhibition_decay = self.config['inhibition_decay']

    cells_shape = [hidden_size]
    batch_cells_shape = [batch_size, hidden_size]

    inhibition = torch.zeros(cells_shape)
    filtered = torch.zeros(batch_cells_shape)

    # Inhibit over time within a batch (because we don't bother having repeats for this).
    for i in range(0, batch_size):
      # Create a mask with a 1 for this batch only
      this_batch_mask = torch.zeros([batch_size, 1])
      this_batch_mask[i][0] = 1.0

      refraction = 1.0 - inhibition
      refraction_2d = refraction.unsqueeze(0)  # add batch dim
      refracted = torch.abs(encoding) * refraction_2d

      # Find the "winners". The top k elements in each batch sample. this is
      # what top_k does.
      # ---------------------------------------------------------------------
      top_k_mask = utils.build_topk_mask(refracted, dim=-1, k=k)

      # Retrospectively add batch-sparsity per cell: pick the top-k (for now
      # k=1 only). TODO make this allow top N per batch.
      # ---------------------------------------------------------------------
      batch_filtered = encoding * top_k_mask  # apply mask 3 to output 2
      this_batch_filtered = batch_filtered * this_batch_mask

      this_batch_topk = top_k_mask * this_batch_mask
      fired, _ = torch.max(this_batch_topk, dim=0)  # reduce over batch
      inhibition = inhibition * inhibition_decay + fired  # set to 1

      filtered = filtered + this_batch_filtered

    return filtered, inhibition

  def forward(self, inputs):  # pylint: disable=arguments-differ
    inputs = torch.flatten(inputs, start_dim=1)
    inputs = inputs.detach()

    with torch.no_grad():
      encoding = self.layer(inputs)
      filtered_encoding, _ = self.apply_sparse_filter(encoding)

    # Override encoding to become binary mask
    top_k_mask = utils.build_topk_mask(filtered_encoding, dim=-1, k=self.config['sparsity'])

    return top_k_mask.detach()
