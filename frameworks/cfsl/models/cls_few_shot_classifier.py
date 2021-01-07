"""cls_few_shot_classifier.py"""

import os
import random

from collections import OrderedDict, defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torchvision
import matplotlib
import matplotlib.pyplot as plt

from cls_module.cls import CLS

from utils.generic import set_torch_seed, calculate_cosine_distance

matplotlib.use('Agg')


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
    self.output_units = int(self.num_classes_per_set if self.overwrite_classes_in_each_task else \
        (self.num_classes_per_set * self.num_support_sets) / self.class_change_interval)

    print('output units =', self.output_units)

    # Dynamically sets the label learner output units, based on task configuration
    self.cls_config['ltm']['classifier']['output_units'] = [self.output_units, 2000]
    self.cls_config['stm']['classifier']['output_units'] = self.output_units

    self.cls_config['ltm']['num_support_sets'] = self.num_support_sets
    self.cls_config['ltm']['num_support_set_steps'] = self.num_support_set_steps
    self.cls_config['ltm']['num_target_set_steps'] = self.num_target_set_steps

    self.writer = SummaryWriter()
    self.current_iter = 0

    # Build the CLS module
    self.model = CLS(input_shape=self.input_shape, config=self.cls_config, writer=self.writer)

    self.ltm_state_dict = None
    self.ltm_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.model.ltm.vgg_optimizer, T_max=self.total_epochs,
                                                              eta_min=self.cls_config['ltm']['min_learning_rate'])

    # Determine the device to use (CPU, GPU, multi-GPU)
    self.device = torch.device('cpu')

    if torch.cuda.is_available():
      self.device = torch.cuda.current_device()

      if torch.cuda.device_count() > 1:
        self.model = nn.DataParallel(self.model)

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


  def get_replay_batch(self, replay_buffer, x_support, y_support, k=None, interleave=True):
    """Build an interleaved batch with current support set and random samples from the replay buffer."""
    x_replay_set = [x_support]
    y_replay_set = [y_support]

    if interleave and (replay_buffer['inputs'] and replay_buffer['labels']):
      replay_buffer_inputs_flat = torch.cat(replay_buffer['inputs'])
      replay_buffer_labels_flat = torch.cat(replay_buffer['labels'])

      replay_buffer_indices = list(range(replay_buffer_inputs_flat.size(0)))

      if k is None or k > len(replay_buffer_indices):
        k = len(replay_buffer_indices)

      replay_buffer_random_idx = random.sample(replay_buffer_indices, k=k)

      x_replay_set.append(replay_buffer_inputs_flat[replay_buffer_random_idx])
      y_replay_set.append(replay_buffer_labels_flat[replay_buffer_random_idx])

    return torch.cat(x_replay_set), torch.cat(y_replay_set)

  def forward(self, data_batch, training_phase):  # pylint: disable=arguments-differ
    """
    Perform one step using CLS to produce losses and summary statistics.
    :return:
    """
    del training_phase

    x_support_set, x_target_set, y_support_set, y_target_set, *_ = data_batch

    num_study_steps = self.cls_config['study_steps']
    num_consolidation_steps = self.cls_config['consolidation_steps']
    replay_method = 'recall'  # recall, or groundtruth
    replay_interleave = True
    replay_num_samples = 5

    per_task_preds = []

    per_task_target_ltm_loss = []
    per_task_target_ltm_accuracy = []
    per_task_support_ltm_accuracy = []
    per_task_ltm_matching_accuracy = []

    for _, (x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task) in \
            enumerate(zip(x_support_set, y_support_set, x_target_set, y_target_set)):

      self.model.reset(['stm'])
      self.model.ltm.load_state_dict(self.ltm_state_dict)  # Special reset for LTM-VGG

      c, h, w = x_target_set_task.shape[-3:]
      x_target_set_task = x_target_set_task.view(-1, c, h, w).to(self.device)
      y_target_set_task = y_target_set_task.view(-1).to(self.device)

      step_idx = 0

      replay_buffer = {
          'inputs': [],
          'labels': []
      }

      for _, (x_support_set_sub_task, y_support_set_sub_task) in \
            enumerate(zip(x_support_set_task, y_support_set_task)):

        # TODO: Continous STM without forgetting
        self.model.reset(['stm'])

        # in the future try to adapt the features using a relational component
        x_support_set_sub_task = x_support_set_sub_task.view(-1, c, h, w).to(self.device)
        y_support_set_sub_task = y_support_set_sub_task.view(-1).to(self.device)

        # Memorise the support sets in STM
        self.model.ltm.eval()
        with torch.no_grad():
          _, pre_ltm_support_outputs = self.model.ltm(inputs=x_support_set_sub_task,
                                                      targets=None,
                                                      labels=y_support_set_sub_task)

          stm_support_input = pre_ltm_support_outputs['memory']['output'].detach()

        self.model.stm.train()
        for _ in range(num_study_steps):
          self.model.stm(inputs=stm_support_input, targets=x_support_set_sub_task, labels=y_support_set_sub_task)
          step_idx += 1

        # Consolidate replayed inputs into LTM
        self.model.ltm.train()
        self.model.ltm.update_predictor(0)
        for _ in range(num_consolidation_steps):
          x_replay_set, y_replay_set = self.get_replay_batch(
              replay_buffer, x_support_set_sub_task, y_support_set_sub_task,
              k=replay_num_samples, interleave=replay_interleave)

          self.model.ltm(inputs=x_replay_set, targets=None, labels=y_replay_set)
          step_idx += 1

        # Recall support set from STM
        # TODO: Spontaneous recall to retrieve previous support sets
        self.eval()
        with torch.no_grad():
          _, stm_support_outputs = self.model.stm(inputs=stm_support_input,
                                                  targets=x_support_set_sub_task,
                                                  labels=y_support_set_sub_task)

        support_stm_images = stm_support_outputs['memory']['decoding']
        support_stm_preds = stm_support_outputs['classifier']['predictions']
        support_stm_softmax_preds = F.softmax(support_stm_preds, dim=0).argmax(dim=1)

        if replay_method == 'recall':
          replay_buffer['inputs'].append(support_stm_images)
          replay_buffer['labels'].append(support_stm_softmax_preds)
        else:
          replay_buffer['inputs'].append(x_support_set_sub_task)
          replay_buffer['labels'].append(y_support_set_sub_task)

      x_support_set_task = x_support_set_task.view(-1, c, h, w).to(self.device)
      y_support_set_task = y_support_set_task.view(-1).to(self.device)

      self.eval()
      with torch.no_grad():
        _, post_ltm_support_outputs = self.model.ltm(inputs=x_support_set_task,
                                                     targets=None,
                                                     labels=y_support_set_task)
        step_idx += 1

        target_losses, target_outputs = self.model.ltm(inputs=x_target_set_task,
                                                       targets=None,
                                                       labels=y_target_set_task)
        step_idx += 1

      # Compute Matching Accuracy
      # ---------------------------------------------------------------------------------------------------------------
      per_class_embeddings = []
      support_ltm_encodings = post_ltm_support_outputs['memory']['output']

      for i in range(self.output_units):
        temp_class_encodings = torch.zeros((self.num_samples_per_support_class * self.num_classes_per_set,
                                            *support_ltm_encodings.shape[1:]))
        count = 0
        for encoding, y in zip(support_ltm_encodings, y_support_set_task):
          if y == i:
            temp_class_encodings[count] = encoding
            count += 1
        mean_encoding = torch.mean(temp_class_encodings, dim=0)
        per_class_embeddings.append(mean_encoding)

      per_class_embeddings = torch.stack(per_class_embeddings, dim=0)
      per_class_embeddings = per_class_embeddings.view(self.batch_size,
                                                       per_class_embeddings.shape[0],
                                                       np.prod(per_class_embeddings.shape[1:]))

      target_ltm_encodings = target_outputs['memory']['output']
      target_ltm_encodings = target_ltm_encodings.view(self.batch_size,
                                                       target_ltm_encodings.shape[0],
                                                       np.prod(target_ltm_encodings.shape[1:]))

      matching_preds, _ = calculate_cosine_distance(per_class_embeddings, y_support_set_task, target_ltm_encodings)
      matching_preds = matching_preds.view(-1, matching_preds.shape[-1])
      # matching_loss = F.cross_entropy(input=matching_preds, target=y_target_set_task)

      softmax_matching_preds = F.softmax(matching_preds, dim=1).argmax(dim=1)
      matching_accuracy = torch.eq(softmax_matching_preds, y_target_set_task).data.cpu().float().mean()
      per_task_ltm_matching_accuracy.append(matching_accuracy)

      # Measure LTM performance on support set
      # ---------------------------------------------------------------------------------------------------------------
      support_ltm_preds = post_ltm_support_outputs['classifier']['predictions']
      support_ltm_softmax_preds = F.softmax(support_ltm_preds, dim=0).argmax(dim=1)
      support_ltm_accuracy = torch.eq(support_ltm_softmax_preds, y_support_set_task).data.cpu().float().mean()
      per_task_support_ltm_accuracy.append(support_ltm_accuracy)

      # Measure LTM performance on the target set
      # ---------------------------------------------------------------------------------------------------------------
      target_ltm_preds = target_outputs['classifier']['predictions']
      target_ltm_softmax_preds = F.softmax(target_ltm_preds, dim=1).argmax(dim=1)
      target_ltm_accuracy = torch.eq(target_ltm_softmax_preds, y_target_set_task).data.cpu().float().mean()

      per_task_preds.append(target_ltm_preds.detach().cpu().numpy())
      per_task_target_ltm_loss.append(target_losses['memory']['loss'])
      per_task_target_ltm_accuracy.append(target_ltm_accuracy)

    loss_metric_dict = dict()

    # These are already logged as the main metrics:  `loss` and `accuracy`
    # loss_metric_dict['target_ltm_loss'] = per_task_target_ltm_loss
    # loss_metric_dict['target_ltm_accuracy'] = per_task_target_ltm_accuracy

    loss_metric_dict['support_ltm_accuracy'] = per_task_support_ltm_accuracy
    loss_metric_dict['matching_ltm_accuracy'] = per_task_ltm_matching_accuracy

    losses = self.get_across_task_loss_metrics(total_losses=per_task_target_ltm_loss,
                                               total_accuracies=per_task_target_ltm_accuracy,
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

    *_, x, y = data_batch

    x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1]).to(self.device)
    y = y.view(-1).to(self.device).long()

    # Train the unsupervised LTM
    self.model.ltm.update_predictor(1)
    model_losses, outputs = self.model.pretrain(inputs=x, labels=y)
    preds = outputs['ltm']['memory']['predictions'].detach()

    softmax_preds = F.softmax(preds, dim=1).argmax(dim=1)
    accuracy = torch.eq(softmax_preds, y).data.cpu().float().mean()

    losses = dict()
    losses['loss'] = model_losses['ltm']['memory']['loss']
    losses['accuracy'] = accuracy
    losses['learning_rate'] = self.ltm_scheduler.get_last_lr()[0]

    self.current_iter += 1

    # Update the LTM state dict
    self.ltm_state_dict = self.model.ltm.get_state_dict()

    self.ltm_scheduler.step()
    self.zero_grad()

    return losses, None

  def run_validation_iter(self, data_batch):
    """
    Runs an outer loop evaluation step on the meta-model's parameters.
    :param data_batch: input data batch containing the support set and target set input, output pairs
    :param epoch: the index of the current epoch
    :return: The losses of the ran iteration.
    """
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
    net_mutated = dict()

    for key in net:
      if key.startswith('model.ltm.classifier') or key.startswith('model.stm'):
        continue

      net_mutated[key] = net[key]

    state['network'] = OrderedDict(net_mutated)
    state_dict_loaded = state['network']

    self.load_state_dict(state_dict=state_dict_loaded, strict=False)
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
