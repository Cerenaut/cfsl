import torch
from torchvision import datasets, transforms

from omniglot_one_shot_dataset import OmniglotTransformation, OmniglotOneShotDataset

from cls_module.memory.ltm.visual_component import VisualComponent
from cls_module.memory.stm.aha import AHA

config = {
    "ltm": {
        "learning_rate": 0.005,

        "filters": 121,
        "kernel_size": 10,
        "stride": 5,
        "eval_stride": 1,
        "padding": 4,

        "encoder_nonlinearity": "none",
        "decoder_nonlinearity": "sigmoid",

        "use_bias": True,
        "use_tied_weights": True,
        "use_lifetime_sparsity": True,

        "sparsity": 1,
        "sparsity_output_factor": 4.0,

        "output_pool_size": 4,
        "output_pool_stride": 4,
        "output_norm_per_sample": True,
        "output_shape": [1, 121, 13, 13],
    },
    "stm": {
        "ps": {
            "inhibition_decay": 0.95,
            "knockout_rate": 0.25,
            "init_scale": 10.0,
            "num_units": 225,
            "sparsity": 10
        },
        "pc": {},
        "pr": {
            "learning_rate": 0.01,
            "weight_decay": 0.000025,

            "num_units": 800,
            "input_dropout": 0.25,
            "hidden_dropout": 0.0,

            "encoder_nonlinearity": "leaky_relu",
            "decoder_nonlinearity": "none",
            "use_bias": True,

            "train_with_noise": 0.05,
            "train_with_noise_pp": 0.005,
            "test_with_noise": 0.0,
            "test_with_noise_pp": 0.0,
            "sparsity": 10,
            "sparsity_boost": 1.5,
            "sparsen": False,
            "softmax": False,
            "gain": 1.0,
            "sum_norm": 10.0
        },
        "pm": {
            "learning_rate": 0.01,
            "weight_decay": 0.00004,

            "num_units": 100,
            "input_dropout": 0.0,
            "hidden_dropout": 0.0,

            "encoder_nonlinearity": "leaky_relu",
            "decoder_nonlinearity": "sigmoid",
            "use_bias": True
        }
    }
}

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
