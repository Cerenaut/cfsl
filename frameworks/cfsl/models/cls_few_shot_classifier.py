"""cls_few_shot_classifier.py"""

import os
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from cls_module import CLS

from utils.generic import set_torch_seed, calculate_cosine_distance


class CLSFewShotClassifier(nn.Module):
  """Few shot classifier based on CLS module."""

  def __init__(self, **kwargs):
    """
    Initializes a MAML few shot learning system
    :param im_shape: The images input size, in batch, c, h, w shape
    :param device: The device to use to use the model on.
    :param args: A namedtuple of arguments specifying various hyperparameters.
    """
    super(CLSFewShotClassifier, self).__init__()

    for key, value in kwargs.items():
      setattr(self, key, value)

    self.input_shape = (2, self.image_channels, self.image_height, self.image_width)
    self.current_epoch = -1
    self.rng = set_torch_seed(seed=self.seed)

    print(self.input_shape)
    self.classifier = CLSMOdule(input_shape=self.input_shape)

    self.device = torch.device('cpu')

    if torch.cuda.is_available():
      self.device = torch.cuda.current_device()
      if torch.cuda.device_count() > 1:
        self.classifier = nn.DataParallel(self.classifier)


    print("Outer Loop parameters")
    num_params = 0
    for name, param in self.named_parameters():
      if param.requires_grad:
        print(name, param.shape)


    print("Memory parameters")
    num_params = 0
    for name, param in self.trainable_names_parameters(exclude_params_with_string=None):
      if param.requires_grad:
        print(name, param.shape)
        product = 1
        for item in param.shape:
          product = product * item
        num_params += product
    print('Total Memory parameters', num_params)

    self.optimizer = optim.Adam(self.trainable_parameters(exclude_list=[]),
                                lr=self.meta_learning_rate,
                                weight_decay=self.weight_decay, amsgrad=False)

    self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.total_epochs,
                                                          eta_min=self.min_learning_rate)
    self.to(self.device)


  def trainable_names_parameters(self, exclude_params_with_string=None):
    """
    Returns an iterator over the trainable parameters of the model.
    """
    for name, param in self.named_parameters():
      if exclude_params_with_string is not None:
        if param.requires_grad and all(
            list([exclude_string not in name for exclude_string in exclude_params_with_string])):
          yield (name, param)
      else:
        if param.requires_grad:
          yield (name, param)

  def forward(self, data_batch, training_phase):
    """
    Builds tf graph for Matching Networks, produces losses and summary statistics.
    :return:
    """

    data_batch = [item.to(self.device) for item in data_batch]

    x_support_set, x_target_set, y_support_set, y_target_set, _, _ = data_batch

    # print('support 0 shape =', x_support_set.shape, x_target_set.shape, y_support_set.shape, y_target_set.shape)

    x_support_set = x_support_set.view(-1, x_support_set.shape[-3], x_support_set.shape[-2],
                                        x_support_set.shape[-1])
    x_target_set = x_target_set.view(-1, x_target_set.shape[-3], x_target_set.shape[-2], x_target_set.shape[-1])
    y_support_set = y_support_set.view(-1)
    y_target_set = y_target_set.view(-1)

    output_units = self.num_classes_per_set if self.overwrite_classes_in_each_task else \
        self.num_classes_per_set * self.num_support_sets

    print('output units =', output_units)

    y_support_set_one_hot = int_to_one_hot(y_support_set)

    g_encoded_images = []

    h, w, c = x_support_set.shape[-3:]

    x_support_set = x_support_set.view(size=(self.batch_size, -1, h, w, c))
    x_target_set = x_target_set.view(size=(self.batch_size, -1, h, w, c))
    y_support_set = y_support_set.view(size=(self.batch_size, -1))
    y_target_set = y_target_set.view(self.batch_size, -1)

    print('data shape =', x_support_set.shape, x_target_set.shape, y_support_set.shape, y_target_set.shape)
    print(y_support_set)
    count = 0
    for x_support_set_task, y_support_set_task in zip(x_support_set,
                                                      y_support_set):  # produce embeddings for support set images
      count += 1
      support_set_cnn_embed, _ = self.classifier.forward(x=x_support_set_task)  # nsc * nc, h, w, c

      per_class_embeddings = torch.zeros(
          (output_units, int(np.prod(support_set_cnn_embed.shape) / (self.num_classes_per_set
                                                                      * support_set_cnn_embed.shape[-1])),
            support_set_cnn_embed.shape[-1])).to(x_support_set_task.device)

      counter_dict = defaultdict(lambda: 0)

      print(support_set_cnn_embed.shape)
      for x, y in zip(support_set_cnn_embed, y_support_set_task):
        counter_dict[y % output_units] += 1
        print(y % output_units, counter_dict[y % output_units] - 1, x.shape, x.mean())
        per_class_embeddings[y % output_units][counter_dict[y % output_units] - 1] = x
      print(per_class_embeddings.shape)
      per_class_embeddings = per_class_embeddings.mean(1)
      g_encoded_images.append(per_class_embeddings)

    f_encoded_image, _ = self.classifier.forward(x=x_target_set.view(-1, h, w, c))
    f_encoded_image = f_encoded_image.view(self.batch_size, -1, f_encoded_image.shape[-1])
    print('f_encoded_image', f_encoded_image.shape)
    g_encoded_images = torch.stack(g_encoded_images, dim=0)
    print('g_encoded', g_encoded_images.shape)

    preds, similarities = calculate_cosine_distance(support_set_embeddings=g_encoded_images,
                                                    support_set_labels=y_support_set_one_hot.float(),
                                                    target_set_embeddings=f_encoded_image)

    y_target_set = y_target_set.view(-1)
    preds = preds.view(-1, preds.shape[-1])
    loss = F.cross_entropy(input=preds, target=y_target_set)

    softmax_target_preds = F.softmax(preds, dim=1).argmax(dim=1)
    accuracy = torch.eq(softmax_target_preds, y_target_set).data.cpu().float().mean()
    losses = dict()
    losses['loss'] = loss
    losses['accuracy'] = accuracy

    return losses, preds.view(self.batch_size,
                              self.num_support_sets * self.num_classes_per_set *
                              self.num_samples_per_target_class,
                              output_units)

  def trainable_parameters(self, exclude_list):
    """
    Returns an iterator over the trainable parameters of the model.
    """
    for name, param in self.named_parameters():
        if all([entry not in name for entry in exclude_list]):
            if param.requires_grad:
              yield param

  def trainable_named_parameters(self, exclude_list):
    """
    Returns an iterator over the trainable parameters of the model.
    """
    for name, param in self.named_parameters():
      if all([entry not in name for entry in exclude_list]):
        if param.requires_grad:
          yield name, param

  def train_forward_prop(self, data_batch, epoch, current_iter):
    """
    Runs an outer loop forward prop using the meta-model and base-model.
    :param data_batch: A data batch containing the support set and the target set input, output pairs.
    :param epoch: The index of the currrent epoch.
    :return: A dictionary of losses for the current step.
    """
    losses, per_task_preds = self.forward(data_batch=data_batch, training_phase=True)
    return losses, per_task_preds.detach().cpu().numpy()

  def evaluation_forward_prop(self, data_batch, epoch):
    """
    Runs an outer loop evaluation forward prop using the meta-model and base-model.
    :param data_batch: A data batch containing the support set and the target set input, output pairs.
    :param epoch: The index of the currrent epoch.
    :return: A dictionary of losses for the current step.
    """
    losses, per_task_preds = self.forward(data_batch=data_batch, training_phase=False)

    return losses, per_task_preds.detach().cpu().numpy()

  def meta_update(self, loss):
    """
    Applies an outer loop update on the meta-parameters of the model.
    :param loss: The current crossentropy loss.
    """
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

  def run_train_iter(self, data_batch, epoch, current_iter):
    """
    Runs an outer loop update step on the meta-model's parameters.
    :param data_batch: input data batch containing the support set and target set input, output pairs
    :param epoch: the index of the current epoch
    :return: The losses of the ran iteration.
    """
    epoch = int(epoch)
    self.scheduler.step(epoch=epoch)
    if self.current_epoch != epoch:
      self.current_epoch = epoch
      # print(epoch, self.optimizer)

    if not self.training:
      self.train()

    losses, per_task_preds = self.train_forward_prop(data_batch=data_batch, epoch=epoch, current_iter=current_iter)

    self.meta_update(loss=losses['loss'])
    losses['learning_rate'] = self.scheduler.get_lr()[0]
    self.zero_grad()

    return losses, per_task_preds

  def run_validation_iter(self, data_batch):
    """
    Runs an outer loop evaluation step on the meta-model's parameters.
    :param data_batch: input data batch containing the support set and target set input, output pairs
    :param epoch: the index of the current epoch
    :return: The losses of the ran iteration.
    """

    if self.training:
        self.eval()

    losses, per_task_preds = self.evaluation_forward_prop(data_batch=data_batch, epoch=self.current_epoch)

    return losses, per_task_preds

  def save_model(self, model_save_dir, state):
    """
    Save the network parameter state and experiment state dictionary.
    :param model_save_dir: The directory to store the state at.
    :param state: The state containing the experiment state and the network. It's in the form of a dictionary
    object.
    """
    state['network'] = self.state_dict()
    torch.save(state, f=model_save_dir)

  def load_model(self, model_save_dir, model_name, model_idx):
    """
    Load checkpoint and return the state dictionary containing the network state params and experiment state.
    :param model_save_dir: The directory from which to load the files.
    :param model_name: The model_name to be loaded from the direcotry.
    :param model_idx: The index of the model (i.e. epoch number or 'latest' for the latest saved model of the current
    experiment)
    :return: A dictionary containing the experiment state and the saved model parameters.
    """
    filepath = os.path.join(model_save_dir, "{}_{}".format(model_name, model_idx))

    state = torch.load(filepath, map_location='cpu')
    net = dict(state['network'])

    state['network'] = OrderedDict(net)
    state_dict_loaded = state['network']
    self.load_state_dict(state_dict=state_dict_loaded)
    self.starting_iter = state['current_iter']

    return state

