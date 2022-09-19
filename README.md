# Continual few-shot learning

## Requirements
- PyTorch 1.5.1+
    - Follow instructions here to set it up locally (depends on your environment)

## Getting Started
First, you need to setup the CLS module before using it with any of the available frameworks.

1. Change into the `cls_module` directory
2. Execute the `python setup.py develop` command to install the package and its dependencies

## Frameworks

### Omniglot Lake Benchmark
This is an implementation of the one-shot generalization benchmark introduced by Lake. The code is available under the
directory `frameworks/lake`.

To run an experiment using the Lake framework, you will need a valid configuration file. There is an existing configuration
file located in `frameworks/lake/config.json` with the default configuration.

Run the experiment using `python oneshot_cls.py --config path/to/config.json`


### CFSL Benchmark
The code is available under `frameworks/cfsl` and is derived from https://github.com/AntreasAntoniou/FewShotContinualLearning

To run the experiments with CLS, you can simply modify the configuration file in `omniglot_cls.json`
and then run the experiment using `bash omniglot_cls.sh GPU_ID latest`.

**Note:** Set `GPU_ID` to `0` if you are not using a GPU, and `1` if you are using a GPU.
