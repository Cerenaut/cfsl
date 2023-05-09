"""Data Transformations."""

import torch
import numpy as np

class NoiseTransformation:
  def __init__(self, minval=0., noise_type='sp_float', noise_factor=0.2):
    self.minval = minval
    self.noise_type = noise_type
    self.noise_factor = noise_factor

  def __call__(self, image):
    
    image_shape = image.shape
    image_size = np.prod(image_shape)

    if self.noise_type == 'sp_float' or self.noise_type == 'sp_binary':
      noise_mask = np.zeros(image_size, dtype=np.float32)
      noise_mask[:int(self.noise_factor * image_size)] = 1
      np.random.shuffle(noise_mask)

      noise_mask = np.reshape(noise_mask, [image_shape[0], image_shape[1], image_shape[2]])

      noise_image = np.random.uniform(self.minval, 1.0, size=image_shape)

      if self.noise_type == 'sp_binary':
        noise_image = np.sign(noise_image)

      noise_image = np.multiply(noise_image, noise_mask, dtype=np.float32)  # retain noise in positions of noise mask
      noise_image = torch.from_numpy(noise_image)

      image = image * (1 - noise_mask)  # zero out noise positions
      corrupted_image = image + noise_image  # add in the noise
      corrupted_image
    else:
      if self.noise_type == 'none':
        raise RuntimeWarning("Add noise has been called despite noise_type of 'none'.")
      else:
        raise NotImplementedError("The noise_type '{0}' is not supported.".format(self.noise_type))

    return corrupted_image

class OcclusionTransformation:
  def __init__(self, degrade_type='circle', degrade_value=0, degrade_factor=0.5, random_value=0.0):
    self.degrade_type = degrade_type
    self.degrade_value = degrade_value
    self.degrade_factor = degrade_factor
    self.random_value = random_value

  def degrade_image_shape(self, image):
    """
    :param degrade_value:
    :param rect_size: radius expressed as proportion of image (half height or width for rectangle)
    :param shape_type: rect or circle
    :return:
    """
    rect_size = self.degrade_factor
    shape_type = self.degrade_type
    degrade_value = self.degrade_value

    image_shape = image.shape
    image_size = np.prod(image_shape)

    height = image_shape[1]
    width = image_shape[2]

    r = int(rect_size * height) # expressed as pixels (choose height, assume square)

    # random start position
    xs = np.random.uniform(low=r, high=width-r, size=[1]).astype(np.int64)
    ys = np.random.uniform(low=r, high=height-r, size=[1]).astype(np.int64)

    int_image = np.arange(0, image_size)
    col = np.int64(int_image % width)  # same shape as image tensor, but values are the col idx
    row = np.int64(int_image / height)  # same but for row idx

    # col = np.expand_dims(col, axis=0)   # add a batch dimension
    # row = np.expand_dims(row, axis=0)

    if shape_type == 'rect':
      mask_x = np.logical_or((col < xs), (col > xs + 2*r))
      mask_y = np.logical_or((row < ys), (row > ys + 2*r))
      preserve_mask = np.logical_or(mask_x, mask_y)
    elif shape_type == 'circle':
      circle_r = np.square(col - xs) + np.square(row - ys)
      preserve_mask = circle_r > np.square(r)
    else:
      raise RuntimeError("Shape type : {0} not supported.".format(shape_type))

    preserve_mask = np.float32(preserve_mask)
    preserve_mask = np.reshape(preserve_mask, [image_shape[0], image_shape[1], image_shape[2]])
    degraded_image = np.multiply(image, preserve_mask)

    # now set the 'degraded' pixels to chosen value
    degraded_mask = 1.0 - preserve_mask
    degraded_mask_vals = degrade_value * degraded_mask
    degraded_image = degraded_image + degraded_mask_vals

    return degraded_image

  def __call__(self, image):
    # TODO(@abdel): Implement other degradation types
    # if self.degrade_type == 'vertical' or self.degrade_type == 'horizontal':
    #   return self.degrade_image_half(image)
    # elif self.degrade_type == 'random':
    #   return self.degrade_image_random(image)

    if self.degrade_type == 'rect' or self.degrade_type == 'circle':
      return self.degrade_image_shape(image)
    else:
      raise RuntimeError("Degrade type {0} not supported.".format(self.degrade_type))
