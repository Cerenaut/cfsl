"""OmniglotOneShotDataset class."""

import os
import copy

from PIL import Image

import torch
from torch.utils.data import Dataset

import torchvision
from torchvision.datasets.utils import download_and_extract_archive, check_integrity

import numpy as np

import imageio
from scipy import ndimage

import tensorflow as tf
tf.enable_eager_execution()

TF_PREPROCESS = False

def tf_centre_of_mass(images, shape):
  # Input volumes
  b, h, w, c = shape

  y_indices = np.zeros((1, h, 1, 1))
  x_indices = np.zeros((1, 1, w, 1))

  for y in range(h):
    y_indices[0][y][0][0] = y
  for x in range(w):
    x_indices[0][0][x][0] = x

  y_indices = tf.constant(y_indices, dtype=tf.float32)
  x_indices = tf.constant(x_indices, dtype=tf.float32)

  sum_y = tf.reduce_sum(images * y_indices, axis=[1,2,3])
  sum_x = tf.reduce_sum(images * x_indices, axis=[1,2,3])
  sum_p = tf.reduce_sum(images, axis=[1,2,3])

  mean_y = sum_y / sum_p
  mean_x = sum_x / sum_p

  centre_of_mass = tf.concat([mean_x, mean_y], axis=0)
  return centre_of_mass

class OmniglotTransformation:
  """Transform Omniglot digits by resizing, centring mass and inverting background/foreground."""

  def __init__(self, centre=True, invert=True, resize_factor=1.0):
    self.centre = centre
    self.invert = invert
    self.resize_factor = resize_factor

  def __call__(self, x):
    # Resize
    if self.resize_factor != 1.0:
      height = int(self.resize_factor * x.shape[1])
      width = int(self.resize_factor * x.shape[2])

      if TF_PREPROCESS:
        x = x.permute(1, 2, 0)
        x = tf.image.resize_images(tf.convert_to_tensor(x.numpy()), [height, width]).numpy()
      else:
        x = torchvision.transforms.ToPILImage()(x)
        x = torchvision.transforms.functional.resize(x, size=[height, width])

      x = torchvision.transforms.functional.to_tensor(x)

    # Invert image
    if self.invert:
      x = torch.max(x) - x

    # Centre the image
    if self.centre:
      # NCHW => NHWC
      x = x.permute(1, 2, 0)

      # Compute centre
      centre = np.array([int(x.shape[0]) * 0.5, int(x.shape[1]) * 0.5])

      # Compute centre of mass
      if TF_PREPROCESS:
        x = tf.convert_to_tensor(x.numpy())
        centre_of_mass = tf_centre_of_mass([x], [1, int(x.shape[0]), int(x.shape[1]), 1])
      else:
        centre_of_mass = ndimage.measurements.center_of_mass(x.numpy())
        centre_of_mass = np.array(centre_of_mass[:-1])

      # Compute translation
      if TF_PREPROCESS:
        translation = centre - centre_of_mass  # e.g. CoM = 24, centre = 26 => 26 - 24 = 2
      else:
        translation = (centre - centre_of_mass).tolist()
        translation.reverse()

      # Apply transformation
      if TF_PREPROCESS:
        x = tf.contrib.image.translate([x], [translation], interpolation='BILINEAR')
        x = tf.squeeze(x, axis=0).numpy()  # Drop the batch dimension
      else:
        # NHWC => NCHW
        x = x.permute(2, 0, 1)
        x = torchvision.transforms.ToPILImage()(x)
        x = torchvision.transforms.functional.affine(x, 0, translation, scale=1.0, shear=0, resample=Image.BILINEAR)

      # Convert back to tensor
      x = torchvision.transforms.functional.to_tensor(x)

    return x


class OmniglotOneShotDataset(Dataset):
  """Face Landmarks dataset."""

  num_runs = 20
  fname_label = 'class_labels.txt'

  folder = 'omniglot_oneshot'
  download_url_prefix = 'https://github.com/brendenlake/omniglot/raw/master/python/one-shot-classification'
  zips_md5 = {
      'all_runs': 'e8996daecdf12afeeb4a53a179f06b19'
  }

  def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
    """
    Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    self.root = root
    self.train = train
    self.transform = transform
    self.target_transform = target_transform

    self.root = os.path.join(root, self.folder)
    self.target_folder = self._get_target_folder()
    self.phase_folder = self._get_phase_folder()

    if download:
      self.download()

    self.filenames, self.labels = self.get_filenames_and_labels()

    if not self._check_integrity():
      raise RuntimeError('Dataset not found or corrupted.' +
                         ' You can use download=True to download it')

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    label = self.labels[idx]

    image_path = self.filenames[idx]
    image = imageio.imread(image_path)

    # Convert to float values in [0, 1
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min())

    if self.transform:
      image = self.transform(image)

    if self.target_transform:
      label = self.target_transform(label)

    return image, label

  def _check_integrity(self):
    zip_filename = self._get_target_folder()
    if not check_integrity(os.path.join(self.root, zip_filename + '.zip'), self.zips_md5[zip_filename]):
      return False
    return True

  def download(self):
    if self._check_integrity():
      print('Files already downloaded and verified')
      return

    filename = self._get_target_folder()
    zip_filename = filename + '.zip'
    url = self.download_url_prefix + '/' + zip_filename
    download_and_extract_archive(url, self.root,
                                 extract_root=os.path.join(self.root, filename),
                                 filename=zip_filename, md5=self.zips_md5[filename])

  def get_filenames_and_labels(self):
    filenames = []
    labels = []

    for r in range(1, self.num_runs + 1):
      rs = str(r)
      if len(rs) == 1:
        rs = '0' + rs

      run_folder = 'run' + rs
      target_path = os.path.join(self.root, self.target_folder)
      # run_path = os.path.join(target_path, run_folder)

      with open(os.path.join(target_path, run_folder, self.fname_label)) as f:
        content = f.read().splitlines()
      pairs = [line.split() for line in content]

      test_files = [pair[0] for pair in pairs]
      train_files = [pair[1] for pair in pairs]

      train_labels = list(range(self.num_runs))
      test_labels = copy.copy(train_labels)      # same labels as train, because we'll read them in this order

      test_files = [os.path.join(target_path, file) for file in test_files]
      train_files = [os.path.join(target_path, file) for file in train_files]

      if self.train:
        filenames.extend(train_files)
        labels.extend(train_labels)
      else:
        filenames.extend(test_files)
        labels.extend(test_labels)

    return filenames, labels

  def _get_target_folder(self):
    return 'all_runs'

  def _get_phase_folder(self):
    return 'training' if self.train else 'test'
