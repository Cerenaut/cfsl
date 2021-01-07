"""VisualComponent class."""

import torch
import torch.optim as optim
import torch.nn.functional as F

from cls_module import utils

from cls_module.memory.interface import MemoryInterface
from meta_neural_network_architectures import VGGActivationNormNetwork

class VGG(MemoryInterface):
  """An implementation of a long-term memory module using sparse convolutional autoencoder."""

  global_key = 'ltm'
  local_key = 'vgg'

  def build(self):
    """Build Visual Component as long-term memory module."""
    self.predictor_idx = 0

    self.num_support_sets = self.config['num_support_sets']
    self.num_support_set_steps = self.config['num_support_set_steps']
    self.num_target_set_steps = self.config['num_target_set_steps']

    model = VGGActivationNormNetwork(input_shape=self.input_shape,
                                     num_output_classes=self.config['classifier']['output_units'],
                                     num_stages=self.config['num_stages'],
                                     use_channel_wise_attention=self.config['use_channel_wise_attention'],
                                     num_filters=self.config['num_filters'],
                                     num_support_set_steps=2 * self.num_support_sets * self.num_support_set_steps,
                                     num_target_set_steps=self.num_target_set_steps + 1).to(self.device)

    model_optimizer = optim.AdamW(model.parameters(),
                                  lr=self.config['meta_learning_rate'],
                                  weight_decay=self.config['weight_decay'],
                                  amsgrad=False)

    self.add_module(self.local_key, model)
    self.add_optimizer(self.local_key, model_optimizer)

    # Compute expected output shape
    with torch.no_grad():
      sample_input = torch.rand(1, *(self.input_shape[1:])).to(self.device)
      _, sample_output = model(sample_input, num_step=0, return_features=True)
      sample_output = self.prepare_encoding(sample_output)
      self.output_shape = list(sample_output.data.shape)
      self.output_shape[0] = -1

  def get_state_dict(self):
    return {
        'model_state_dict': self.vgg.state_dict(),
        'optimizer_state_dict': self.vgg_optimizer.state_dict(),
    }

  def load_state_dict(self, state_dict):
    if state_dict is None:
      state_dict = self.get_state_dict()
    self.vgg.load_state_dict(state_dict['model_state_dict'])
    self.vgg_optimizer.load_state_dict(state_dict['optimizer_state_dict'])

  def update_predictor(self, idx):
    predictor_name = 'layer_dict.linear_'

    if idx == self.predictor_idx:
      return

    prev_predictor = predictor_name + str(self.predictor_idx)
    next_predictor = predictor_name + str(idx)

    for name, p in self.vgg.named_parameters():
      if name.startswith(prev_predictor):
        p.requires_grad = False
      elif name.startswith(next_predictor):
        p.requires_grad = True

    # Update predictor index
    self.predictor_idx = idx

  def forward_memory(self, inputs, targets, labels):
    """Perform an optimization step using the memory module."""
    del targets  # Supervised learning via labels

    if self.vgg.training:
      self.vgg_optimizer.zero_grad()

    preds, encoding = self.vgg.forward(x=inputs, num_step=0, return_features=True)

    loss = F.cross_entropy(input=preds[self.predictor_idx], target=labels)

    if self.vgg.training:
      loss.backward()
      self.vgg_optimizer.step()

    output_encoding = self.prepare_encoding(encoding)

    outputs = {
        'encoding': encoding,
        'predictions': preds[self.predictor_idx],

        'output': output_encoding  # Designated output for linked modules
    }

    self.features = {
        'vgg': output_encoding.detach().cpu()
    }

    return loss, outputs

  def forward(self, inputs, targets, labels=None):  # pylint: disable=arguments-differ
    """Perform an optimisation step with the entire CLS module."""
    losses = {}
    outputs = {}

    memory_loss, memory_outputs = self.forward_memory(inputs, targets, labels)

    losses['memory'] = {
        'loss': memory_loss
    }

    outputs['memory'] = memory_outputs

    outputs['classifier'] = {
        'loss': memory_loss,
        'predictions': memory_outputs['predictions']
    }

    self.step += 1

    return losses, outputs

  def prepare_encoding(self, encoding):
    """Postprocessing for the encoding."""
    encoding = encoding.detach()

    return encoding
