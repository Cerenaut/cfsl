import json

import torch
from torchvision import datasets, transforms

from omniglot_one_shot_dataset import OmniglotTransformation, OmniglotOneShotDataset

from cls_module.memory.ltm.visual_component import VisualComponent
from cls_module.memory.stm.aha import AHA

with open('aha_config.json') as config_file:
  config = json.load(config_file)

batch_size = 20
image_tfms = transforms.Compose([
    transforms.ToTensor(),
    OmniglotTransformation(resize_factor=0.5)
])

study_loader = torch.utils.data.DataLoader(
    OmniglotOneShotDataset('./data', train=True, download=True,
                           transform=image_tfms, target_transform=None),
    batch_size=batch_size, shuffle=False)

x, _ = next(iter(study_loader))

ltm = VisualComponent(config=config['ltm'], input_shape=x.shape, target_shape=None)
ltm.eval()
_, outputs = ltm(x, targets=x)
encoding = outputs['memory']['output']

stm = AHA(config=config['stm'], input_shape=encoding.shape, target_shape=x.shape)

stm.train()
stm(inputs=encoding, targets=x, labels=None)

stm.eval()
stm(inputs=encoding, targets=x, labels=None)
