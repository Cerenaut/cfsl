"""utils.py"""

import os
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
