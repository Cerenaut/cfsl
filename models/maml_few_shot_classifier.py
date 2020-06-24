import os
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import AdamW

from meta_neural_network_architectures import VGGActivationNormNetwork, \
    VGGActivationNormNetworkWithAttention
from meta_optimizer import LSLRGradientDescentLearningRule
from pytorch_utils import int_to_one_hot
from standard_neural_network_architectures import TaskRelationalEmbedding, \
    SqueezeExciteDenseNetEmbeddingSmallNetwork, CriticNetwork, VGGEmbeddingNetwork

from utils.generic import set_torch_seed, calculate_cosine_distance


class MAMLFewShotClassifier(nn.Module):
    def __init__(self, batch_size, seed, num_classes_per_set, num_samples_per_support_class,
                 num_samples_per_target_class, image_channels,
                 num_filters, num_blocks_per_stage, num_stages, dropout_rate, output_spatial_dimensionality,
                 image_height, image_width, num_support_set_steps, init_learning_rate, num_target_set_steps,
                 conditional_information, min_learning_rate, total_epochs, weight_decay, meta_learning_rate, **kwargs):
        """
        Initializes a MAML few shot learning system
        :param im_shape: The images input size, in batch, c, h, w shape
        :param device: The device to use to use the model on.
        :param args: A namedtuple of arguments specifying various hyperparameters.
        """
        super(MAMLFewShotClassifier, self).__init__()
        self.batch_size = batch_size
        self.current_epoch = -1
        self.rng = set_torch_seed(seed=seed)
        self.num_classes_per_set = num_classes_per_set
        self.num_samples_per_support_class = num_samples_per_support_class
        self.num_samples_per_target_class = num_samples_per_target_class
        self.image_channels = image_channels
        self.num_filters = num_filters
        self.num_blocks_per_stage = num_blocks_per_stage
        self.num_stages = num_stages
        self.dropout_rate = dropout_rate
        self.output_spatial_dimensionality = output_spatial_dimensionality
        self.image_height = image_height
        self.image_width = image_width
        self.num_support_set_steps = num_support_set_steps
        self.init_learning_rate = init_learning_rate
        self.num_target_set_steps = num_target_set_steps
        self.conditional_information = conditional_information
        self.min_learning_rate = min_learning_rate
        self.total_epochs = total_epochs
        self.weight_decay = weight_decay
        self.meta_learning_rate = meta_learning_rate

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.device = torch.device('cpu')

        if torch.cuda.is_available():
          self.device = torch.cuda.current_device()

        self.clip_grads = True
        self.rng = set_torch_seed(seed=seed)
        self.build_module()

    def build_module(self):
        return NotImplementedError

    def setup_optimizer(self):

        exclude_param_string = None if "none" in self.exclude_param_string else self.exclude_param_string
        self.optimizer = optim.Adam(self.trainable_parameters(exclude_params_with_string=exclude_param_string),
                                    lr=0.001,
                                    weight_decay=self.weight_decay, amsgrad=False)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                              T_max=self.total_epochs,
                                                              eta_min=0.001)
        print('min learning rate'.self.min_learning_rate)
        self.to(self.device)

        print("Inner Loop parameters")
        num_params = 0
        for key, value in self.inner_loop_optimizer.named_parameters():
            print(key, value.shape)
            num_params += np.prod(value.shape)
        print('Total inner loop parameters', num_params)

        print("Outer Loop parameters")
        num_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.shape)
                num_params += np.prod(value.shape)
        print('Total outer loop parameters', num_params)

        print("Memory parameters")
        num_params = 0
        for name, param in self.get_params_that_include_strings(included_strings=['classifier']):
            if param.requires_grad:
                print(name, param.shape)
                num_params += np.prod(value.shape)
        print('Total Memory parameters', num_params)

    def get_params_that_include_strings(self, included_strings, include_all=False):
        for name, param in self.named_parameters():
            if any([included_string in name for included_string in included_strings]) and not include_all:
                yield name, param
            if all([included_string in name for included_string in included_strings]) and include_all:
                yield name, param

    def get_per_step_loss_importance_vector(self):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        """
        loss_weights = np.ones(shape=(self.number_of_training_steps_per_iter)) * (
                1.0 / self.number_of_training_steps_per_iter)
        decay_rate = 1.0 / self.number_of_training_steps_per_iter / self.multi_step_loss_num_epochs
        min_value_for_non_final_losses = self.minimum_per_task_contribution / self.number_of_training_steps_per_iter
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (self.current_epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (self.current_epoch * (self.number_of_training_steps_per_iter - 1) * decay_rate),
            1.0 - ((self.number_of_training_steps_per_iter - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights

    def apply_inner_loop_update(self, loss, names_weights_copy, use_second_order, current_step_idx):
        """
        Applies an inner loop update given current step's loss, the weights to update, a flag indicating whether to use
        second order derivatives and the current step's index.
        :param loss: Current step's loss with respect to the support set.
        :param names_weights_copy: A dictionary with names to parameters to update.
        :param use_second_order: A boolean flag of whether to use second order derivatives.
        :param current_step_idx: Current step's index.
        :return: A dictionary with the updated weights (name, param)
        """
        self.classifier.zero_grad(params=names_weights_copy)
        grads = torch.autograd.grad(loss, names_weights_copy.values(),
                                    create_graph=use_second_order, allow_unused=True)
        names_grads_copy = dict(zip(names_weights_copy.keys(), grads))

        for key, grad in names_grads_copy.items():
            if grad is None:
                print('NOT FOUND INNER LOOP', key)

        names_weights_copy = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                     names_grads_wrt_params_dict=names_grads_copy,
                                                                     num_step=current_step_idx)

        return names_weights_copy

    def get_inner_loop_parameter_dict(self, params, exclude_strings=None):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        param_dict = dict()

        if exclude_strings is None:
            exclude_strings = []

        for name, param in params:
            if param.requires_grad:
                if all([item not in name for item in exclude_strings]):
                    if "norm_layer" not in name and 'bn' not in name and 'prelu' not in name:
                        param_dict[name] = param.to(device=self.device)
        return param_dict

    def net_forward(self, x, y, weights, backup_running_statistics, training, num_step,
                    return_features=False):
        """
        A base model forward pass on some data points x. Using the parameters in the weights dictionary. Also requires
        boolean flags indicating whether to reset the running statistics at the end of the run (if at evaluation phase).
        A flag indicating whether this is the training session and an int indicating the current step's number in the
        inner loop.
        :param x: A data batch of shape b, c, h, w
        :param y: A data targets batch of shape b, n_classes
        :param weights: A dictionary containing the weights to pass to the network.
        :param backup_running_statistics: A flag indicating whether to reset the batch norm running statistics to their
         previous values after the run (only for evaluation)
        :param training: A flag indicating whether the current process phase is a training or evaluation.
        :param num_step: An integer indicating the number of the step in the inner loop.
        :return: the crossentropy losses with respect to the given y, the predictions of the base model.
        """

        if return_features:
            preds, features = self.classifier.forward(x=x, params=weights,
                                                      training=training,
                                                      backup_running_statistics=backup_running_statistics,
                                                      num_step=num_step,
                                                      return_features=return_features)

            loss = F.cross_entropy(preds, y)

            return loss, preds, features


        else:
            preds = self.classifier.forward(x=x, params=weights,
                                            training=training,
                                            backup_running_statistics=backup_running_statistics,
                                            num_step=num_step)

            loss = F.cross_entropy(preds, y)

            return loss, preds

    def trainable_parameters(self, exclude_params_with_string=None):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for name, param in self.named_parameters():
            if exclude_params_with_string is not None:
                if param.requires_grad and all(
                        list([exclude_string not in name for exclude_string in exclude_params_with_string])):
                    yield param
            else:
                if param.requires_grad:
                    yield param

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

    def train_forward_prop(self, data_batch, epoch):
        """
        Runs an outer loop forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_preds = self.forward(data_batch=data_batch, epoch=epoch,
                                              use_second_order=self.second_order and
                                                               epoch > self.first_order_to_second_order_epoch,
                                              use_multi_step_loss_optimization=self.use_multi_step_loss_optimization,
                                              num_steps=self.number_of_training_steps_per_iter,
                                              training_phase=True)
        return losses, per_task_preds

    def evaluation_forward_prop(self, data_batch, epoch):
        """
        Runs an outer loop evaluation forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_preds = self.forward(data_batch=data_batch, epoch=epoch, use_second_order=False,
                                              use_multi_step_loss_optimization=self.use_multi_step_loss_optimization,
                                              num_steps=self.number_of_evaluation_steps_per_iter,
                                              training_phase=False)

        return losses, per_task_preds

    def meta_update(self, loss, exclude_string_list=None, retain_graph=False):
        """
        Applies an outer loop update on the meta-parameters of the model.
        :param loss: The current crossentropy loss.
        """
        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if 'imagenet' in self.dataset_name:
            for name, param in self.trainable_names_parameters(exclude_params_with_string=exclude_string_list):
                #

                if self.clip_grads and param.grad is None and param.requires_grad:
                    print(name, 'no grad information computed')
                # else:
                #     print("passed", name)
                else:
                    if param.grad is None:
                        print('no grad information computed', name)
                # print('No Grad', name, param.shape)
                if self.clip_grads and param.grad is not None and param.requires_grad and "softmax":
                    param.grad.data.clamp_(-10, 10)

        self.optimizer.step()
