"""SparseAutoencoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from cls_module.utils import activation_fn, build_topk_mask, truncated_normal_

class SparseAutoencoder(nn.Module):
  """Standard EC with decoder layers removed and maxpool added.
  """

  def __init__(self, input_shape, config, output_shape=None):
    super(SparseAutoencoder, self).__init__()

    self.input_shape = list(input_shape)
    self.input_size = np.prod(self.input_shape[1:])
    self.config = config

    if output_shape is None:
      self.output_shape = self.input_shape
      self.output_size = self.input_size
    else:
      self.output_shape = list(output_shape)
      self.output_size = np.prod(self.output_shape[1:])

    self.input_shape[0] = -1
    self.output_shape[0] = -1

    self.encoder = self.build_encoder()
    self.decoder = self.build_decoder(self.encoder, tied_weights=self.config['use_tied_weights'])

    self.encoder_nonlinearity = activation_fn(self.config['encoder_nonlinearity'])
    self.decoder_nonlinearity = activation_fn(self.config['decoder_nonlinearity'])

  def initialize(self, m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
      torch.nn.init.xavier_uniform_(m.weight)
      m.weight.data = truncated_normal_(m.weight.data, std=0.03)
      if m.bias is not None:
        torch.nn.init.zeros_(m.bias)

  def build_encoder(self):
    encoder = nn.Conv2d(self.input_shape[1], self.config['filters'],
                        kernel_size=self.config['kernel_size'],
                        stride=self.config['stride'],
                        bias=self.config['use_bias'],
                        padding=self.config['padding'])
    self.initialize(encoder)

    return encoder

  def build_decoder(self, encoder, tied_weights=False):
    """Build a decoder using ConvTranspose2d with optional tied weights with the encoder."""
    decoder = nn.ConvTranspose2d(self.config['filters'], self.input_shape[1],
                                 kernel_size=self.config['kernel_size'],
                                 stride=self.config['stride'],
                                 bias=self.config['use_bias'],
                                 padding=self.config['padding'])
    self.initialize(decoder)

    if not tied_weights:
      return decoder

    return lambda x: F.conv_transpose2d(x, encoder.weight,
                                        bias=decoder.bias.data,
                                        stride=self.config['stride'],
                                        padding=self.config['padding'])

  def encode(self, inputs):
    encoding = self.encoder(inputs)
    encoding = self.encoder_nonlinearity(encoding)

    return encoding

  def filter(self, encoding):
    """Build filtering/masking for specified encoding."""
    encoding_nhwc = encoding.permute(0, 2, 3, 1)  # NCHW => NHWC

    top_k_input = encoding_nhwc

    # Find the "winners". The top k elements in each batch sample. this is
    # what top_k does.
    # ---------------------------------------------------------------------
    k = int(self.config['sparsity'])

    if not self.training:
      k = int(k * self.config['sparsity_output_factor'])

    top_k_mask = build_topk_mask(top_k_input, dim=-1, k=k)

    # Retrospectively add batch-sparsity per cell: pick the top-k (for now
    # k=1 only). TODO make this allow top N per batch.
    # Note: Not working.
    # ---------------------------------------------------------------------
    if self.training and self.config['use_lifetime_sparsity']:
      # torch.max() does not currently support specifying dims like TensorFlow
      # This alternative method works similarly to tf.reduce_max(.., axis=[0, 1, 2], ..)
      def max_reduce(x, dim=0, keepdim=True):
        values, _ = x.view(x.size(0) * x.size(1) * x.size(2), -1).max(dim, keepdim=keepdim)
        if keepdim:
          values = torch.reshape(values, [1, 1, 1, -1])
        return values

      batch_max = max_reduce(top_k_input, keepdim=True)  # input shape: batch,cells, output shape: cells
      batch_mask_bool = top_k_input >= batch_max # inhibit cells (mask=0) until spike has decayed
      batch_mask = batch_mask_bool.float()

      either_mask = torch.max(top_k_mask, batch_mask) # logical OR, i.e. top-k or top-1 per cell in batch
    else:
      either_mask = top_k_mask

    filtered_encoding_nhwc = encoding_nhwc * either_mask  # Apply mask 3 to output 2
    filtered_encoding = filtered_encoding_nhwc.permute(0, 3, 1, 2)  # NHWC => NCHW

    return filtered_encoding

  def decode(self, encoding):
    decoding = self.decoder(encoding)
    decoding = self.decoder_nonlinearity(decoding)
    decoding = torch.reshape(decoding, self.output_shape)

    return decoding

  def forward(self, x):  # pylint: disable=arguments-differ
    encoding = self.encode(x)

    if self.config['sparsity'] > 0:
      encoding = self.filter(encoding)

    decoding = self.decode(encoding)

    return encoding, decoding
