"""FastNN module."""

import torch
import torch.nn as nn

import numpy as np

from cls_module.utils import activation_fn, initialize_parameters

class SimpleAutoencoder(nn.Module):
  """A simple encoder-decoder network."""

  def __init__(self, input_shape, config, output_shape=None):
    super(SimpleAutoencoder, self).__init__()

    self.input_shape = list(input_shape)
    self.input_size = np.prod(self.input_shape[1:])

    if output_shape is None:
      self.output_shape = self.input_shape
      self.output_size = self.input_size
    else:
      self.output_shape = list(output_shape)
      self.output_size = np.prod(self.output_shape[1:])

    self.input_shape[0] = -1
    self.output_shape[0] = -1

    self.config = config

    self.build()

  def build(self):
    """Build the network architecture."""
    self.encoder = nn.Linear(self.input_size, self.config['num_units'], bias=self.config['use_bias'])
    self.decoder = nn.Linear(self.config['num_units'], self.output_size, bias=self.config['use_bias'])

    self.encoder_nonlinearity = activation_fn(self.config['encoder_nonlinearity'])
    self.decoder_nonlinearity = activation_fn(self.config['decoder_nonlinearity'])

    self.encoder_dropout = nn.Dropout(p=self.config['dropout'])

    self.reset_parameters()

  def reset_parameters(self):
    self.apply(lambda m: initialize_parameters(m, weight_init='xavier_normal_', bias_init='zeros_'))

  def encode(self, inputs):
    inputs = torch.flatten(inputs, start_dim=1)
    inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())

    encoding = self.encoder(inputs)
    encoding = self.encoder_nonlinearity(encoding)
    encoding = self.encoder_dropout(encoding)

    return encoding

  def decode(self, encoding):
    decoding = self.decoder(encoding)
    decoding = self.decoder_nonlinearity(decoding)

    decoding = torch.reshape(decoding, self.output_shape)

    return decoding

  def forward(self, x):  # pylint: disable=arguments-differ
    encoding = self.encode(x)
    decoding = self.decode(encoding)

    return encoding, decoding
