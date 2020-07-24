"""utils.py"""

import os
import math
import random
import datetime

import torch
import torch.nn as nn

import numpy as np


def activation_fn(fn_type):
  """Simple switcher for choosing activation functions."""
  if fn_type == 'none':
    fn = lambda x: x
  elif fn_type == 'relu':
    fn = nn.ReLU()
  elif fn_type in ['leaky-relu', 'leaky_relu']:
    fn = nn.LeakyReLU()
  elif fn_type == 'tanh':
    fn = nn.Tanh()
  elif fn_type == 'sigmoid':
    fn = nn.Sigmoid()
  elif fn_type == 'softmax':
    fn = nn.Softmax()
  else:
    raise NotImplementedError(
        'Activation function implemented: ' + str(fn_type))

  return fn


def build_topk_mask(x, dim=1, k=2):
  """
  Simple functional version of KWinnersMask/KWinners since
  autograd function apparently not currently exportable by JIT
  """
  res = torch.zeros_like(x)
  _, indices = torch.topk(x, k=k, dim=dim, sorted=False)
  return res.scatter(dim, indices, 1)


def truncated_normal_(tensor, mean=0, std=1):
  size = tensor.shape
  tmp = tensor.new_empty(size + (4,)).normal_()
  valid = (tmp < 2) & (tmp > -2)
  ind = valid.max(-1, keepdim=True)[1]
  tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
  tensor.data.mul_(std).add_(mean)
  return tensor


def initialize_parameters(m, weight_init='xavier_uniform_', bias_init='zeros_'):
  """Initialize nn.Module parameters."""
  if not isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
    return

  weight_init_fn = get_initializer_by_name(weight_init)

  if m.weight is not None and weight_init_fn is not None:
    weight_init_fn(m.weight)

  bias_init_fn = get_initializer_by_name(bias_init)

  if m.bias is not None:
    bias_init_fn(m.bias)


def get_initializer_by_name(init_type):
  # Handle custom initializers
  if init_type == 'truncated_normal_':
    return lambda x: truncated_normal_(x, mean=0.0, std=0.03)

  return getattr(torch.nn.init, init_type, None)


def reduce_max(x, dim=0, keepdim=False):
  """
  Performs `torch.max` over multiple dimensions of `x`
  """
  axes = sorted(dim)
  maxed = x
  for axis in reversed(axes):
    maxed, _ = maxed.max(axis, keepdim)
  return maxed

def get_top_k(x, k, mask_type="pass_through", topk_dim=0, scatter_dim=0):
  """Finds the top k values in a tensor, returns them as a tensor.

  Accepts a tensor as input and returns a tensor of the same size. Values
  in the top k values are preserved or converted to 1, remaining values are
  floored to 0 or -1.

      Example:
          >>> a = torch.tensor([1, 2, 3])
          >>> k = 1
          >>> ans = get_top_k(a, k)
          >>> ans
          torch.tensor([0, 0, 3])

  Args:
      x: (tensor) input.
      k: (int) how many top k examples to return.
      mask_type: (string) Options: ['pass_through', 'hopfield', 'binary']
      topk_dim: (int) Which axis do you want to grab topk over? ie. batch = 0
      scatter_dim: (int) Make it the same as topk_dim to scatter the values
  """

  # Initialize zeros matrix
  zeros = torch.zeros_like(x)

  # find top k vals, indicies
  vals, idx = torch.topk(x, k, dim=topk_dim)

  # Scatter vals onto zeros
  top_ks = zeros.scatter(scatter_dim, idx, vals)

  if mask_type != "pass_through":
    # pass_through does not convert any values.

    if mask_type == "binary":
      # Converts values to 0, 1
      top_ks[top_ks > 0.] = 1.
      top_ks[top_ks < 1.] = 0.

    elif mask_type == "hopfield":
      # Converts values to -1, 1
      top_ks[top_ks >= 0.] = 1.
      top_ks[top_ks < 1.] = -1.

    else:
      raise Exception('Valid options: "pass_through", "hopfield" (-1, 1), or "binary" (0, 1)')

  return top_ks

def add_image_noise_flat(image, label=None, minval=0., noise_type='sp_binary', noise_factor=0.2):
  """If the image is flat (batch, size) then use this version. It reshapes and calls the add_imagie_noise()"""
  image_shape = image.shape.as_list()
  image = tf.reshape(image, (-1, image_shape[1], 1, 1))
  image = add_image_noise(image, label, minval, noise_type, noise_factor)
  image = tf.reshape(image, (-1, image_shape[1]))
  return image


def add_image_noise(image, label=None, minval=0., noise_type='sp_binary', noise_factor=0.2):
  image_shape = image.shape.as_list()
  image_size = np.prod(image_shape[1:])

  if noise_type == 'sp_float' or noise_type == 'sp_binary':
    noise_mask = np.zeros(image_size)
    noise_mask[:int(noise_factor * image_size)] = 1
    noise_mask = tf.convert_to_tensor(noise_mask, dtype=tf.float32)
    noise_mask = tf.random_shuffle(noise_mask)
    noise_mask = tf.reshape(noise_mask, [-1, image_shape[1], image_shape[2], image_shape[3]])

    noise_image = tf.random_uniform(image_shape, minval, 1.0)
    if noise_type == 'sp_binary':
      noise_image = tf.sign(noise_image)
    noise_image = tf.multiply(noise_image, noise_mask)  # retain noise in positions of noise mask

    image = tf.multiply(image, (1 - noise_mask))  # zero out noise positions
    corrupted_image = image + noise_image  # add in the noise
  else:
    if noise_type == 'none':
      raise RuntimeWarning("Add noise has been called despite noise_type of 'none'.")
    else:
      raise NotImplementedError("The noise_type '{0}' is not supported.".format(noise_type))

  if label is None:
    return corrupted_image

  return corrupted_image, label


def add_image_salt_noise_flat(image, label=None, noise_val=0., noise_factor=0., mode='add'):
  """If the image is flat (batch, size) then use this version. It reshapes and calls the add_image_noise()"""
  image_shape = image.shape
  image = torch.reshape(image, (-1, image_shape[1], 1, 1))
  image = add_image_salt_noise(image, label, noise_val, noise_factor, mode)
  image = torch.reshape(image, (-1, image_shape[1]))
  return image

def add_image_salt_pepper_noise_flat(image, label=None, salt_val=1., pepper_val=0., noise_factor=0.):
  """If the image is flat (batch, size) then use this version. It reshapes and calls the add_image_noise()"""
  image_shape = image.shape
  image = torch.reshape(image, (-1, image_shape[1], 1, 1))
  image = add_image_salt_noise(image, label, salt_val, noise_factor, 'replace')
  image = add_image_salt_noise(image, label, pepper_val, noise_factor, 'replace')
  image = torch.reshape(image, (-1, image_shape[1]))
  return image

def add_image_salt_noise(image, label=None, noise_val=0., noise_factor=0., mode='add'):
  """ Add salt noise.

  :param image:
  :param label:
  :param noise_val: value of 'salt' (can be +ve or -ve, must be non zero to have an effect)
  :param noise_factor: the proportion of the image
  :param mode: 'replace' = replace existing value, 'add' = noise adds to the existing value
  :return:
  """

  device = image.device

  image_shape = image.shape
  image_size = np.prod(image_shape[1:])

  # random shuffle of chosen number of active bits
  noise_mask = np.zeros(image_size, dtype=np.float32)
  noise_mask[:int(noise_factor * image_size)] = 1
  np.random.shuffle(noise_mask)

  noise_mask = np.reshape(noise_mask, [-1, image_shape[1], image_shape[2], image_shape[3]])
  noise_mask = torch.from_numpy(noise_mask).to(device)

  if mode == 'replace':
    image = image * (1 - noise_mask)  # image: zero out noise positions

  image = image + (noise_mask * noise_val)  # image: add in the noise at the chosen value

  if label is None:
    return image

  return image, label


def square_image_shape_from_1d(filters):
  """
  Make 1d tensor as square as possible. If the length is a prime, the worst case, it will remain 1d.
  Assumes and retains first dimension as batches.
  """
  height = int(math.sqrt(filters))

  while height > 1:
    width_remainder = filters % height
    if width_remainder == 0:
      break
    else:
      height = height - 1

  width = filters // height
  area = height * width
  lost_pixels = filters - area

  shape = [-1, height, width, 1]

  return shape, lost_pixels
