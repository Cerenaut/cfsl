"""cls_few_shot_classifier.py"""

import os
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from cls_module.cls import CLS

from utils.generic import set_torch_seed, calculate_cosine_distance


class CLSFewShotClassifier(nn.Module):
  """Few shot classifier based on CLS module."""

  def __init__(self, **kwargs):
    """
    Initializes a CLS few shot learning system
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

    # Specify the output units based on the CFSL task parameters
    output_units = self.num_classes_per_set if self.overwrite_classes_in_each_task else \
        self.num_classes_per_set * self.num_support_sets

    self.cls_config['ltm']['classifier']['output_units'] = output_units

    print('output units =', output_units)

    self.writer = SummaryWriter()
    self.current_iter = 0

    # Build the CLS module
    self.classifier = CLS(input_shape=self.input_shape, config=self.cls_config, writer=self.writer)

    # Determine the device to use (CPU, GPU, multi-GPU)
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

  def forward(self, data_batch, training_phase):  # pylint: disable=arguments-differ
    """
    Perform one step using CLS to produce losses and summary statistics.
    :return:
    """
    if self.current_iter == 1:
      exit()

    self.classifier.reset()

    x_support_set, x_target_set, y_support_set, y_target_set, x, y = data_batch

    x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1]).to(self.device)
    y = y.view(-1).to(self.device).long()

    num_pretrain_steps = 240
    num_study_steps = 120

    # Pretrain the LTM-VC
    if training_phase:
      for _ in range(num_pretrain_steps):
        self.classifier(inputs=x, labels=None, mode='pretrain')

    total_per_step_losses = []
    total_per_step_accuracies = []

    per_task_preds = []

    pre_target_loss_update_loss = []
    pre_target_loss_update_acc = []
    post_target_loss_update_loss = []
    post_target_loss_update_acc = []

    for _, (x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task) in \
            enumerate(zip(x_support_set, y_support_set, x_target_set, y_target_set)):

      c, h, w = x_target_set_task.shape[-3:]
      x_target_set_task = x_target_set_task.view(-1, c, h, w).to(self.device)
      y_target_set_task = y_target_set_task.view(-1).to(self.device)

      step_idx = 0

      if training_phase:
        self.classifier.train()
      else:
        self.classifier.eval()

      for _, (x_support_set_sub_task, y_support_set_sub_task) in enumerate(zip(x_support_set_task, y_support_set_task)):

        # in the future try to adapt the features using a relational component
        x_support_set_sub_task = x_support_set_sub_task.view(-1, c, h, w).to(self.device)
        y_support_set_sub_task = y_support_set_sub_task.view(-1).to(self.device)

        # Memorise the support sets in STM
        if training_phase:
          for _ in range(num_study_steps):
            losses, _ = self.classifier(inputs=x_support_set_sub_task, labels=y_support_set_sub_task, mode='study')
            step_idx += 1

        # TODO:
        # 1. Spontanious recall to retrieve old samples
        # 2. Interleave current support set with recalled samples
        # 3. Consolidate into the LTM classifier
        # for num_step in range(num_study_steps):
        #   losses, _ = self.classifier(inputs=x_support_set_sub_task, labels=y_support_set_sub_task, mode='recall')

        #   # support_set_preds = self.classifier.ltm.predictions
        #   # support_set_softmax_preds = F.softmax(support_set_preds, dim=1).argmax(dim=1)
        #   # support_set_accuracy = torch.eq(support_set_softmax_preds, y_support_set_sub_task).data.cpu().float().mean()
        #   step_idx += 1

      # Measure performance on the target set
      self.classifier.eval()
      target_losses, _ = self.classifier(inputs=x_target_set_task, labels=y_target_set_task, mode='recall')

      target_set_preds = self.classifier.ltm.predictions
      target_set_loss = target_losses['ltm']  # target_losses['stm']
      step_idx += 1

      post_update_loss, post_update_target_preds = target_set_loss, target_set_preds

      pre_target_loss_update_loss.append(target_set_loss)
      pre_softmax_target_preds = F.softmax(target_set_preds, dim=1).argmax(dim=1)
      pre_update_accuracy = torch.eq(pre_softmax_target_preds, y_target_set_task).data.cpu().float().mean()
      pre_target_loss_update_acc.append(pre_update_accuracy)

      post_target_loss_update_loss.append(post_update_loss)
      post_softmax_target_preds = F.softmax(post_update_target_preds, dim=1).argmax(dim=1)

      post_update_accuracy = torch.eq(post_softmax_target_preds, y_target_set_task).data.cpu().float().mean()
      post_target_loss_update_acc.append(post_update_accuracy)

      post_softmax_target_preds = F.softmax(post_update_target_preds, dim=1).argmax(dim=1)
      post_update_accuracy = torch.eq(post_softmax_target_preds, y_target_set_task).data.cpu().float().mean()
      post_target_loss_update_acc.append(post_update_accuracy)

      self.writer.add_scalar('target_set_loss', target_set_loss, self.current_iter)
      self.writer.add_scalar('target_set_accuracy', post_update_accuracy, self.current_iter)

      loss = target_set_loss

      total_per_step_losses.append(loss)
      total_per_step_accuracies.append(post_update_accuracy)

      per_task_preds.append(post_update_target_preds.detach().cpu().numpy())

    loss_metric_dict = dict()
    loss_metric_dict['pre_target_loss_update_loss'] = post_target_loss_update_loss
    loss_metric_dict['pre_target_loss_update_acc'] = pre_target_loss_update_acc
    loss_metric_dict['post_target_loss_update_loss'] = post_target_loss_update_loss
    loss_metric_dict['post_target_loss_update_acc'] = post_target_loss_update_acc

    losses = self.get_across_task_loss_metrics(total_losses=total_per_step_losses,
                                               total_accuracies=total_per_step_accuracies,
                                               loss_metrics_dict=loss_metric_dict)

    return losses, per_task_preds

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
    del epoch, current_iter

    losses, per_task_preds = self.forward(data_batch=data_batch, training_phase=True)

    return losses, per_task_preds

  def evaluation_forward_prop(self, data_batch, epoch):
    """
    Runs an outer loop evaluation forward prop using the meta-model and base-model.
    :param data_batch: A data batch containing the support set and the target set input, output pairs.
    :param epoch: The index of the currrent epoch.
    :return: A dictionary of losses for the current step.
    """
    del epoch

    losses, per_task_preds = self.forward(data_batch=data_batch, training_phase=False)

    return losses, per_task_preds

  def run_train_iter(self, data_batch, epoch, current_iter):
    """
    Runs an outer loop update step on the meta-model's parameters.
    :param data_batch: input data batch containing the support set and target set input, output pairs
    :param epoch: the index of the current epoch
    :return: The losses of the ran iteration.
    """
    epoch = int(epoch)

    if self.current_epoch != epoch:
      self.current_epoch = epoch

    if not self.training:
      self.train()

    losses, per_task_preds = self.train_forward_prop(data_batch=data_batch, epoch=epoch, current_iter=current_iter)

    self.current_iter += 1

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

  def get_across_task_loss_metrics(self, total_losses, total_accuracies, loss_metrics_dict):
    losses = dict()

    losses['loss'] = torch.mean(torch.stack(total_losses), dim=(0,))

    losses['accuracy'] = torch.mean(torch.stack(total_accuracies), dim=(0,))

    if 'saved_logits' in loss_metrics_dict:
      losses['saved_logits'] = loss_metrics_dict['saved_logits']
      del loss_metrics_dict['saved_logits']

    for name, value in loss_metrics_dict.items():
      losses[name] = torch.stack(value).mean()

    return losses
