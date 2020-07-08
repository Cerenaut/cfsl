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
from models.maml_few_shot_classifier import MAMLFewShotClassifier


class FineTuneFromScratchFewShotClassifier(MAMLFewShotClassifier):
    def __init__(self, batch_size, seed, num_classes_per_set, num_samples_per_support_class, image_channels,
                 num_filters, num_blocks_per_stage, num_stages, dropout_rate, output_spatial_dimensionality,
                 image_height, image_width, num_support_set_steps, init_learning_rate, num_target_set_steps,
                 conditional_information, min_learning_rate, total_epochs, weight_decay, meta_learning_rate,
                 num_samples_per_target_class, **kwargs):
        """
        Initializes a MAML few shot learning system
        :param im_shape: The images input size, in batch, c, h, w shape
        :param device: The device to use to use the model on.
        :param args: A namedtuple of arguments specifying various hyperparameters.
        """
        super(FineTuneFromScratchFewShotClassifier, self).__init__(batch_size, seed, num_classes_per_set,
                                                                   num_samples_per_support_class,
                                                                   num_samples_per_target_class, image_channels,
                                                                   num_filters, num_blocks_per_stage, num_stages,
                                                                   dropout_rate, output_spatial_dimensionality,
                                                                   image_height, image_width, num_support_set_steps,
                                                                   init_learning_rate, num_target_set_steps,
                                                                   conditional_information, min_learning_rate,
                                                                   total_epochs,
                                                                   weight_decay, meta_learning_rate, **kwargs)

        self.batch_size = batch_size
        self.current_epoch = -1
        self.rng = set_torch_seed(seed=seed)
        self.num_classes_per_set = num_classes_per_set
        self.num_samples_per_support_class = num_samples_per_support_class
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
        self.current_epoch = -1

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.rng = set_torch_seed(seed=seed)

    def param_dict_to_vector(self, param_dict):

        param_list = []

        for name, param in param_dict.items():
            param_list.append(param.view(-1, 1))

        param_as_vector = torch.cat(param_list, dim=0)

        return param_as_vector

    def param_vector_to_param_dict(self, param_vector, names_params_dict):

        new_names_params_dict = dict()
        cur_idx = 0
        for name, param in names_params_dict.items():
            new_names_params_dict[name] = param_vector[cur_idx:cur_idx + param.view(-1).shape[0]].view(param.shape)
            cur_idx += param.view(-1).shape[0]

        return new_names_params_dict

    def build_module(self):
        support_set_shape = (
            self.num_classes_per_set * self.num_samples_per_support_class,
            self.image_channels,
            self.image_height, self.image_width)

        target_set_shape = (
            self.num_classes_per_set * self.num_samples_per_target_class,
            self.image_channels,
            self.image_height, self.image_width)

        x_support_set = torch.ones(support_set_shape)
        x_target_set = torch.ones(target_set_shape)

        # task_size = x_target_set.shape[0]
        x_target_set = x_target_set.view(-1, x_target_set.shape[-3], x_target_set.shape[-2], x_target_set.shape[-1])
        x_support_set = x_support_set.view(-1, x_support_set.shape[-3], x_support_set.shape[-2],
                                           x_support_set.shape[-1])

        num_target_samples = x_target_set.shape[0]
        num_support_samples = x_support_set.shape[0]

        output_units = self.num_classes_per_set if self.overwrite_classes_in_each_task else \
            self.num_classes_per_set * self.num_support_sets

        self.current_iter = 0

        self.classifier = VGGActivationNormNetwork(input_shape=torch.cat([x_support_set, x_target_set], dim=0).shape,
                                                   num_output_classes=output_units,
                                                   num_stages=4, use_channel_wise_attention=True,
                                                   num_filters=48,
                                                   num_support_set_steps=2 * self.num_support_sets * self.num_support_set_steps,
                                                   num_target_set_steps=self.num_target_set_steps + 1,
                                                   )

        print("init learning rate", self.init_learning_rate)
        names_weights_copy = self.get_inner_loop_parameter_dict(self.classifier.named_parameters())

        task_name_params = self.get_inner_loop_parameter_dict(self.named_parameters())

        if self.num_target_set_steps > 0:
            self.dense_net_embedding = SqueezeExciteDenseNetEmbeddingSmallNetwork(
                im_shape=torch.cat([x_support_set, x_target_set], dim=0).shape, num_filters=self.num_filters,
                num_blocks_per_stage=self.num_blocks_per_stage,
                num_stages=self.num_stages, average_pool_outputs=False, dropout_rate=self.dropout_rate,
                output_spatial_dimensionality=self.output_spatial_dimensionality, use_channel_wise_attention=True)

            task_features = self.dense_net_embedding.forward(
                x=torch.cat([x_support_set, x_target_set], dim=0), dropout_training=True)
            task_features = task_features.squeeze()
            encoded_x = task_features
            support_set_features = F.avg_pool2d(encoded_x[:num_support_samples], encoded_x.shape[-1]).squeeze()

            preds, penultimate_features_x = self.classifier.forward(x=torch.cat([x_support_set, x_target_set], dim=0),
                                                                    num_step=0, return_features=True)
            if 'task_embedding' in self.conditional_information:
                self.task_relational_network = TaskRelationalEmbedding(input_shape=support_set_features.shape,
                                                                       num_samples_per_support_class=self.num_samples_per_support_class,
                                                                       num_classes_per_set=self.num_classes_per_set)
                relational_encoding_x = self.task_relational_network.forward(x_img=support_set_features)
                relational_embedding_shape = relational_encoding_x.shape
            else:
                self.task_relational_network = None
                relational_embedding_shape = None

            x_support_set_task = F.avg_pool2d(
                encoded_x[:self.num_classes_per_set * (self.num_samples_per_support_class)],
                encoded_x.shape[-1]).squeeze()
            x_target_set_task = F.avg_pool2d(
                encoded_x[self.num_classes_per_set * (self.num_samples_per_support_class):],
                encoded_x.shape[-1]).squeeze()
            x_support_set_classifier_features = F.avg_pool2d(penultimate_features_x[
                                                             :self.num_classes_per_set * (
                                                                 self.num_samples_per_support_class)],
                                                             penultimate_features_x.shape[-2]).squeeze()
            x_target_set_classifier_features = F.avg_pool2d(
                penultimate_features_x[self.num_classes_per_set * (self.num_samples_per_support_class):],
                penultimate_features_x.shape[-2]).squeeze()

            self.critic_network = CriticNetwork(
                task_embedding_shape=relational_embedding_shape,
                num_classes_per_set=self.num_classes_per_set,
                support_set_feature_shape=x_support_set_task.shape,
                target_set_feature_shape=x_target_set_task.shape,
                support_set_classifier_pre_last_features=x_support_set_classifier_features.shape,
                target_set_classifier_pre_last_features=x_target_set_classifier_features.shape,

                num_target_samples=self.num_samples_per_target_class,
                num_support_samples=self.num_samples_per_support_class,
                logit_shape=preds[self.num_classes_per_set * (self.num_samples_per_support_class):].shape,
                support_set_label_shape=(
                    self.num_classes_per_set * (self.num_samples_per_support_class), self.num_classes_per_set),
                conditional_information=self.conditional_information)

        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(
            total_num_inner_loop_steps=2 * (
                    self.num_support_sets * self.num_support_set_steps) + self.num_target_set_steps + 1,
            learnable_learning_rates=self.learnable_learning_rates,
            init_learning_rate=self.init_learning_rate)

        self.inner_loop_optimizer.initialise(names_weights_dict=names_weights_copy)
        print("Inner Loop parameters")
        for key, value in self.inner_loop_optimizer.named_parameters():
            print(key, value.shape)

        print("Outer Loop parameters")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.shape)

        self.exclude_list = ['classifier', 'inner_loop']
        # self.switch_opt_params(exclude_list=self.exclude_list)

        self.device = torch.device('cpu')
        if torch.cuda.is_available():

            if torch.cuda.device_count() > 1:
                self.to(self.device)
                self.dense_net_embedding = nn.DataParallel(module=self.dense_net_embedding)
            else:
                self.to(self.device)

            self.device = torch.cuda.current_device()

    def switch_opt_params(self, exclude_list):
        print("current trainable params")
        for name, param in self.trainable_names_parameters(exclude_params_with_string=exclude_list):
            print(name, param.shape)
        self.optimizer = AdamW(self.trainable_parameters(exclude_list), lr=self.meta_learning_rate,
                               weight_decay=self.weight_decay, amsgrad=False)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.total_epochs,
                                                              eta_min=self.min_learning_rate)

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
        outputs = {"loss": 0., "preds": 0, "features": 0.}
        if return_features:
            outputs['preds'], outputs['features'] = self.classifier.forward(x=x, params=weights,
                                                                            training=training,
                                                                            backup_running_statistics=backup_running_statistics,
                                                                            num_step=num_step,
                                                                            return_features=return_features)
            if type(outputs['preds']) == tuple:
                if len(outputs['preds']) == 2:
                    outputs['preds'] = outputs['preds'][0]

            outputs['loss'] = F.cross_entropy(outputs['preds'], y)


        else:
            outputs['preds'] = self.classifier.forward(x=x, params=weights,
                                                       training=training,
                                                       backup_running_statistics=backup_running_statistics,
                                                       num_step=num_step)

            if type(outputs['preds']) == tuple:
                if len(outputs['preds']) == 2:
                    outputs['preds'] = outputs['preds'][0]

            outputs['loss'] = F.cross_entropy(outputs['preds'], y)

        return outputs

    def get_per_step_loss_importance_vector(self, current_epoch):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        """
        loss_weights = torch.ones(size=(self.number_of_training_steps_per_iter * self.num_support_sets,),
                                  device=self.device) / (
                               self.number_of_training_steps_per_iter * self.num_support_sets)
        early_steps_decay_rate = (1. / (
                self.number_of_training_steps_per_iter * self.num_support_sets)) / 100.

        loss_weights = loss_weights - (early_steps_decay_rate * current_epoch)

        loss_weights = torch.max(input=loss_weights,
                                 other=torch.ones(loss_weights.shape, device=self.device) * 0.001)

        loss_weights[-1] = 1. - torch.sum(loss_weights[:-1])

        return loss_weights

    def forward(self, data_batch, epoch, use_second_order, use_multi_step_loss_optimization, num_steps, training_phase):
        """
        Runs a forward outer loop pass on the batch of tasks using the MAML/++ framework.
        :param data_batch: A data batch containing the support and target sets.
        :param epoch: Current epoch's index
        :param use_second_order: A boolean saying whether to use second order derivatives.
        :param use_multi_step_loss_optimization: Whether to optimize on the outer loop using just the last step's
        target loss (True) or whether to use multi step loss which improves the stability of the system (False)
        :param num_steps: Number of inner loop steps.
        :param training_phase: Whether this is a training phase (True) or an evaluation phase (False)
        :return: A dictionary with the collected losses of the current outer forward propagation.
        """

        import torchvision
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')

        x_support_set, x_target_set, y_support_set, y_target_set, x, y = data_batch

        self.classifier.zero_grad()

        total_per_step_losses = []

        total_per_step_accuracies = []

        per_task_preds = []
        num_losses = 2
        importance_vector = torch.Tensor([1.0 / num_losses for i in range(num_losses)]).to(self.device)
        step_magnitude = (1.0 / num_losses) / self.total_epochs
        current_epoch_step_magnitude = torch.ones(1).to(self.device) * (step_magnitude * (epoch + 1))

        importance_vector[0] = importance_vector[0] - current_epoch_step_magnitude
        importance_vector[1] = importance_vector[1] + current_epoch_step_magnitude

        pre_target_loss_update_loss = []
        pre_target_loss_update_acc = []
        post_target_loss_update_loss = []
        post_target_loss_update_acc = []

        for task_id, (x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task) in \
                enumerate(zip(x_support_set,
                              y_support_set,
                              x_target_set,
                              y_target_set)):

            c, h, w = x_target_set_task.shape[-3:]
            x_target_set_task = x_target_set_task.view(-1, c, h, w).to(self.device)
            y_target_set_task = y_target_set_task.view(-1).to(self.device)
            target_set_per_step_loss = []
            importance_weights = self.get_per_step_loss_importance_vector(current_epoch=self.current_epoch)
            step_idx = 0

            # print('y_target =', y_target_set_task)

            # grid_img = torchvision.utils.make_grid(x_target_set_task)
            # grid_img = grid_img.permute(1, 2, 0)
            # filepath = 'imgs/epoch_' + str(epoch) + '_' + str(self.current_iter) + '_target_' + str(task_id) + '.png'
            # plt.imshow(grid_img, cmap='gray')
            # plt.savefig(filepath, format='png')
            # plt.close()

            for sub_task_id, (x_support_set_sub_task, y_support_set_sub_task) in \
                    enumerate(zip(x_support_set_task,
                                  y_support_set_task)):
                names_weights_copy = self.get_inner_loop_parameter_dict(self.classifier.named_parameters())
                num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
                print('y_support =', y_support_set_task)

                names_weights_copy = {
                    name.replace('module.', ''): value.unsqueeze(0).repeat(
                        [num_devices] + [1 for i in range(len(value.shape))]) for
                    name, value in names_weights_copy.items()}

                # in the future try to adapt the features using a relational component
                x_support_set_sub_task = x_support_set_sub_task.view(-1, c, h, w).to(self.device)
                y_support_set_sub_task = y_support_set_sub_task.view(-1).to(self.device)

                # grid_img = torchvision.utils.make_grid(x_support_set_sub_task)
                # grid_img = grid_img.permute(1, 2, 0)
                # filepath = 'imgs/epoch_' + str(epoch) + '_' + str(self.current_iter) + '_support_' + str(task_id) + '_' + str(sub_task_id) + '.png'
                # plt.imshow(grid_img, cmap='gray')
                # plt.savefig(filepath, format='png')
                # plt.close()

                if self.num_target_set_steps > 0 and 'task_embedding' in self.conditional_information:
                    image_embedding = self.dense_net_embedding.forward(
                        x=torch.cat([x_support_set_sub_task, x_target_set_task], dim=0), dropout_training=True)
                    x_support_set_task_features = image_embedding[:x_support_set_sub_task.shape[0]]
                    x_target_set_task_features = image_embedding[x_support_set_sub_task.shape[0]:]
                    x_support_set_task_features = F.avg_pool2d(x_support_set_task_features,
                                                               x_support_set_task_features.shape[-1]).squeeze()
                    x_target_set_task_features = F.avg_pool2d(x_target_set_task_features,
                                                              x_target_set_task_features.shape[-1]).squeeze()
                    if self.task_relational_network is not None:
                        task_embedding = self.task_relational_network.forward(x_img=x_support_set_task_features)
                    else:
                        task_embedding = None
                else:
                    task_embedding = None

                for num_step in range(self.num_support_set_steps):
                    support_outputs = self.net_forward(x=x_support_set_sub_task,
                                                       y=y_support_set_sub_task,
                                                       weights=names_weights_copy,
                                                       backup_running_statistics=
                                                       True if (num_step == 0) else False,
                                                       training=True,
                                                       num_step=step_idx,
                                                       return_features=True)

                    names_weights_copy = self.apply_inner_loop_update(loss=support_outputs['loss'],
                                                                      names_weights_copy=names_weights_copy,
                                                                      use_second_order=use_second_order,
                                                                      current_step_idx=step_idx)
                    step_idx += 1

                    if self.use_multi_step_loss_optimization:
                        target_outputs = self.net_forward(x=x_target_set_task,
                                                          y=y_target_set_task, weights=names_weights_copy,
                                                          backup_running_statistics=False, training=True,
                                                          num_step=step_idx,
                                                          return_features=True)
                        target_set_per_step_loss.append(target_outputs['loss'])
                        step_idx += 1

            if not self.use_multi_step_loss_optimization:
                target_outputs = self.net_forward(x=x_target_set_task,
                                                  y=y_target_set_task, weights=names_weights_copy,
                                                  backup_running_statistics=False, training=True,
                                                  num_step=step_idx,
                                                  return_features=True)
                target_set_loss = target_outputs['loss']
                step_idx += 1
            else:

                target_set_loss = torch.sum(
                    torch.stack(target_set_per_step_loss, dim=0) * importance_weights)
            # print(target_set_loss, target_set_per_step_loss, importance_weights)

            # if self.save_preds:
            #     if saved_logits_list is None:
            #         saved_logits_list = []
            #
            #     saved_logits_list.extend(target_outputs['preds'])

            for num_step in range(self.num_target_set_steps):
                predicted_loss = self.critic_network.forward(logits=target_outputs['preds'],
                                                             task_embedding=task_embedding)

                names_weights_copy = self.apply_inner_loop_update(loss=predicted_loss,
                                                                  names_weights_copy=names_weights_copy,
                                                                  use_second_order=use_second_order,
                                                                  current_step_idx=step_idx)
                step_idx += 1

            if self.num_target_set_steps > 0:
                post_update_outputs = self.net_forward(
                    x=x_target_set_task,
                    y=y_target_set_task, weights=names_weights_copy,
                    backup_running_statistics=False, training=True,
                    num_step=step_idx,
                    return_features=True)
                post_update_loss, post_update_target_preds, post_updated_target_features = post_update_outputs[
                                                                                               'loss'], \
                                                                                           post_update_outputs[
                                                                                               'preds'], \
                                                                                           post_update_outputs[
                                                                                               'features']
            else:
                post_update_loss, post_update_target_preds, post_updated_target_features = target_set_loss, \
                                                                                           target_outputs['preds'], \
                                                                                           target_outputs[
                                                                                               'features']

            pre_target_loss_update_loss.append(target_set_loss)
            pre_softmax_target_preds = F.softmax(target_outputs['preds'], dim=1).argmax(dim=1)
            pre_update_accuracy = torch.eq(pre_softmax_target_preds, y_target_set_task).data.cpu().float().mean()
            pre_target_loss_update_acc.append(pre_update_accuracy)

            post_target_loss_update_loss.append(post_update_loss)
            post_softmax_target_preds = F.softmax(post_update_target_preds, dim=1).argmax(dim=1)

            post_update_accuracy = torch.eq(post_softmax_target_preds, y_target_set_task).data.cpu().float().mean()
            post_target_loss_update_acc.append(post_update_accuracy)

            post_softmax_target_preds = F.softmax(post_update_target_preds, dim=1).argmax(dim=1)
            post_update_accuracy = torch.eq(post_softmax_target_preds, y_target_set_task).data.cpu().float().mean()
            post_target_loss_update_acc.append(post_update_accuracy)

            loss = target_outputs['loss']  # * importance_vector[0] + post_update_loss * importance_vector[1]

            total_per_step_losses.append(loss)
            total_per_step_accuracies.append(post_update_accuracy)

            per_task_preds.append(post_update_target_preds.detach().cpu().numpy())

            if not training_phase:
                self.classifier.restore_backup_stats()

        loss_metric_dict = dict()
        loss_metric_dict['pre_target_loss_update_loss'] = post_target_loss_update_loss
        loss_metric_dict['pre_target_loss_update_acc'] = pre_target_loss_update_acc
        loss_metric_dict['post_target_loss_update_loss'] = post_target_loss_update_loss
        loss_metric_dict['post_target_loss_update_acc'] = post_target_loss_update_acc

        losses = self.get_across_task_loss_metrics(total_losses=total_per_step_losses,
                                                   total_accuracies=total_per_step_accuracies,
                                                   loss_metrics_dict=loss_metric_dict)

        return losses, per_task_preds

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

    def run_train_iter(self, data_batch, epoch, current_iter):
        """
        Runs an outer loop update step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """
        epoch = int(epoch)
        # self.scheduler.step(epoch=epoch)

        if self.current_epoch != epoch:
            self.current_epoch = epoch

        if not self.training:
            self.train()

        losses, per_task_preds = self.train_forward_prop(data_batch=data_batch, epoch=epoch)
        exclude_string = None

        # self.meta_update(loss=losses['loss'], exclude_string_list=exclude_string)
        # losses['learning_rate'] = self.scheduler.get_lr()[0]
        self.zero_grad()

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

    def get_across_task_loss_metrics(self, total_losses, total_accuracies, loss_metrics_dict):
        losses = dict()

        losses['loss'] = torch.mean(torch.stack(total_losses), dim=(0,))

        losses['accuracy'] = torch.mean(torch.stack(total_accuracies), dim=(0,))

        if 'saved_logits' in loss_metrics_dict:
            losses['saved_logits'] = loss_metrics_dict['saved_logits']
            del loss_metrics_dict['saved_logits']

        for name, value in loss_metrics_dict.items():
            losses[name] = torch.stack(value).mean()

        for idx_num_step, (name, learning_rate_num_step) in enumerate(self.inner_loop_optimizer.named_parameters()):
            for idx, learning_rate in enumerate(learning_rate_num_step.mean().view(1)):
                losses['task_learning_rate_num_step_{}_{}'.format(idx_num_step,
                                                                  name)] = learning_rate.detach().cpu().numpy()

        return losses
