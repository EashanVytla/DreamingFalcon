def attrdict_monkeypatch_fix():
    import collections
    import collections.abc
    for type_name in collections.abc.__all__:
            setattr(collections, type_name, getattr(collections.abc, type_name))
attrdict_monkeypatch_fix()

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

import numpy as np

import yaml
from attrdict import AttrDict

def horizontal_forward(network, x, y=None, input_shape=(-1,), output_shape=(-1,)):
    batch_with_horizon_shape = x.shape[: -len(input_shape)]
    if not batch_with_horizon_shape:
        batch_with_horizon_shape = (1,)
    if y is not None:
        x = torch.cat((x, y), -1)
        input_shape = (x.shape[-1],)  #
    x = x.reshape(-1, *input_shape)
    x = network(x)

    x = x.reshape(*batch_with_horizon_shape, *output_shape)
    return x

def get_dist(state, argmax=False):
        """
        get_dist takes in a stochastic state and returns a pytorch distribution

        :param state: stochastic state after logit function
        :return: pytorch distribution of stochastic state rep
        """ 

        logit = state.float()
        dist = td.Independent(OneHotDist(logit), 1)
        return dist

def create_stoch(x, stoch, classes, unimix):
    """
    create_stoch converts output vector of representation model
    or transition model into a categorical stochastic rep.

    :param x: output layer (to be used as input to this function)
    :param stoch: from config yaml 
    :param classes: from config yaml
    :param unimix: from config yaml (config.unimix)
    :return: categorical stochastic rep logit
    """ 
    logit = x.view(*x.shape[:-1], stoch, classes)

    # Check if unimix is True
    if unimix:
        # Softmax along the last dimension
        probs = torch.nn.functional.softmax(logit, dim=-1)
        
        # Uniform distribution
        uniform = torch.ones_like(probs) / probs.shape[-1]
        
        # Combine unimix and uniform distributions
        probs = (1 - unimix) * probs + unimix * uniform
        
        # Compute logit from probabilities
        logit = torch.log(probs)

    return logit

def build_network(input_size, hidden_size, num_layers, activation, output_size):
    assert num_layers >= 2, "num_layers must be at least 2"
    activation = getattr(nn, activation)()
    layers = []
    layers.append(nn.Linear(input_size, hidden_size))
    layers.append(activation)

    for i in range(num_layers - 2):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(activation)

    layers.append(nn.Linear(hidden_size, output_size))

    network = nn.Sequential(*layers)
    network.apply(initialize_weights)
    return network


def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

def compute_lambda_values(rewards, values, continues, horizon_length, device, lambda_):
    """
    rewards : (batch_size, time_step, hidden_size)
    values : (batch_size, time_step, hidden_size)
    continue flag will be added
    """
    rewards = rewards[:, :-1]
    continues = continues[:, :-1]
    next_values = values[:, 1:]
    last = next_values[:, -1]
    inputs = rewards + continues * next_values * (1 - lambda_)

    outputs = []
    # single step
    for index in reversed(range(horizon_length - 1)):
        last = inputs[:, index] + continues[:, index] * lambda_ * last
        outputs.append(last)
    returns = torch.stack(list(reversed(outputs)), dim=1).to(device)
    return returns

class OneHotDist(td.OneHotCategorical):
    def __init__(self, logits=None, probs=None, dtype=torch.float32):
        super().__init__(logits=logits, probs=probs, dtype=dtype)

    def sample(self, sample_shape=()):
        if not isinstance(sample_shape, (list, tuple)):
            sample_shape = (sample_shape,)
        logits = self.logits_parameter().to(self.dtype)
        shape = logits.shape
        logits = logits.view([-1, shape[-1]])
        indices = torch.multinomial(F.softmax(logits, dim=-1), np.prod(sample_shape), replacement=True)
        sample = F.one_hot(indices, shape[-1]).float()
        if np.prod(sample_shape) != 1:
            sample = sample.permute((1, 0, 2))
        sample = sample.reshape(sample_shape + shape)
        # Straight through biased gradient estimator.
        probs = self._pad(super().probs_parameter(), sample.shape)
        sample += probs - probs.detach()
        return sample

    def _pad(self, tensor, shape):
        while len(tensor.shape) < len(shape):
            tensor = tensor.unsqueeze(0)
        return tensor

class DynamicInfos:
    def __init__(self, device):
        self.device = device
        self.data = {}

    def append(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)

    def get_stacked(self, time_axis=1):
        stacked_data = AttrDict(
            {
                key: torch.stack(self.data[key], dim=time_axis).to(self.device)
                for key in self.data
            }
        )
        self.clear()
        return stacked_data

    def clear(self):
        self.data = {}


def find_file(file_name):
    cur_dir = os.getcwd()

    for root, dirs, files in os.walk(cur_dir):
        if file_name in files:
            return os.path.join(root, file_name)

    raise FileNotFoundError(
        f"File '{file_name}' not found in subdirectories of {cur_dir}"
    )


def get_base_directory():
    return "/".join(find_file("main.py").split("/")[:-1])


def load_config(config_path):
    if not config_path.endswith(".yml"):
        config_path += ".yml"
    config_path = find_file(config_path)
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return AttrDict(config)

class OneHotDist(td.OneHotCategorical):
    def __init__(self, logits=None, probs=None, dtype=torch.float32):
        super().__init__(logits=logits, probs=probs, dtype=dtype)

    def sample(self, sample_shape=()):
        if not isinstance(sample_shape, (list, tuple)):
            sample_shape = (sample_shape,)
        logits = self.logits_parameter().to(self.dtype)
        shape = logits.shape
        logits = logits.view([-1, shape[-1]])
        indices = torch.multinomial(F.softmax(logits, dim=-1), prod(sample_shape), replacement=True)
        sample = F.one_hot(indices, shape[-1]).float()
        if prod(sample_shape) != 1:
            sample = sample.permute((1, 0, 2))
        sample = sample.reshape(sample_shape + shape)
        # Straight through biased gradient estimator.
        probs = self._pad(super().probs_parameter(), sample.shape)
        sample += probs - probs.detach()
        return sample

    def _pad(self, tensor, shape):
        while len(tensor.shape) < len(shape):
            tensor = tensor.unsqueeze(0)
        return tensor