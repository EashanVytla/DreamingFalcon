import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from torch.distributions import Normal
import numpy as np

from utils import build_network, horizontal_forward


class RSSM(nn.Module):
    def __init__(self, action_size, config):
        super().__init__()
        self.config = config.parameters.dreamer.rssm

        self.recurrent_model = RecurrentModel(action_size, config)
        self.transition_model = TransitionModel(config)
        self.representation_model = RepresentationModel(config)

    def recurrent_model_input_init(self, batch_size):
        return self.transition_model.input_init(
            batch_size
        ), self.recurrent_model.input_init(batch_size)
    
    def create_stoch(self, x):
        """
        create_stoch converts output vector of representation model
        or transition model into a categorical stochastic rep.

        :param x: output layer (to be used as input to this function)
        :return: categorical stochastic rep logit
        """ 
        logit = x.view(*x.shape[:-1], self.config.stoch, self.config.classes)

        # Check if unimix is True
        if self.config._unimix:
            # Softmax along the last dimension
            probs = torch.nn.functional.softmax(logit, dim=-1)
            
            # Uniform distribution
            uniform = torch.ones_like(probs) / probs.shape[-1]
            
            # Combine unimix and uniform distributions
            probs = (1 - self._unimix) * probs + self._unimix * uniform
            
            # Compute logit from probabilities
            logit = torch.log(probs)

        return logit


    def get_dist(self, state, argmax=False):
        """
        get_dist takes in a stochastic state and returns a pytorch distribution

        :param state: stochastic state after logit function
        :return: pytorch distribution of stochastic state rep
        """ 

        logit = state.float()
        dist = td.Independent(OneHotDist(logit), 1)
        return dist
    
    def img_step(self, prev_state, prev_action):
        """
        img_step computes the prior stochastic state (z-hat) and deterministic hidden state (h_t) 
        based on the prev_state and prev_action

        :param prev_state: previous deterministic & stochastic hidden state (in a dictionary)
        :param prev_action: last action
        :return: prior state dictionary {'stoch': z-hat value, 'deter': h_t value}
        """ 
        x = self.recurrent_model(prev_state, prev_action)
        dist = self.get_dist(self.create_stoch(x))
        stoch = dist.sample()
        prior = {'stoch': stoch, 'deter': x}
        return prior
    
    def obj_step(self, prev_state, prev_action, embed):
        """
        obj_step computes the posterior stochastic state (z) based on the prev_state and prev_action

        :param prev_state: previous deterministic & stochastic hidden state (in a dictionary)
        :param prev_action: last action
        :return: prior state dictionary {'stoch': z-hat value, 'deter': h_t value}
        """ 
        prior = self.img_step(prev_state, prev_action)
        x = torch.concat([prior['deter'], embed], -1)

        post = {'stoch': 0, 'deter': 0}
        post['stoch'], post['deter'] = self.representation_model(x)
        return post

class RecurrentModel(nn.Module):
    def __init__(self, action_size, config):
        super().__init__()
        self.config = config.parameters.dreamer.rssm.recurrent_model
        self.device = config.operation.device
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        self.activation = getattr(nn, self.config.activation)()

        #TODO: There should be more than one linear layer in this
        self.linear = nn.Linear(
            self.stochastic_size + action_size, self.config.hidden_size
        )
        self.recurrent = nn.GRUCell(self.config.hidden_size, self.deterministic_size)

    def forward(self, prev_state, action):
        shape = prev_state["stoch"].shape[:-2] + [self.config._stoch * self.config._classes]
        prev_state["stoch"] = prev_state["stoch"].reshape(shape)

        if len(action.shape) > len(prev_state["stoch"].shape):  # 2D actions.
            shape = action.shape[:-2] + [np.prod(action.shape[-2:])]
            action = action.reshape(shape)

        x = torch.cat((prev_state["stoch"], action), 1)
        x = self.activation(self.linear(x))
        x = self.recurrent(x, prev_state["deter"])
        return x

    def input_init(self, batch_size):
        return torch.zeros(batch_size, self.deterministic_size).to(self.device)


class TransitionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.parameters.dreamer.rssm.transition_model
        self.device = config.operation.device
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        self.network = build_network(
            self.deterministic_size,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
            self.stochastic_size * 2,
        )

    def forward(self, x):
        x = self.network(x)
        prior_dist = create_normal_dist(x, min_std=self.config.min_std)
        prior = prior_dist.rsample()
        return prior_dist, prior

    def input_init(self, batch_size):
        return torch.zeros(batch_size, self.stochastic_size).to(self.device)

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

class RepresentationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.parameters.dreamer.rssm.representation_model
        self.embedded_state_size = config.parameters.dreamer.embedded_state_size
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        self.network = build_network(
            self.embedded_state_size + self.deterministic_size,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
            self.stochastic_size * 2,
        )

    #TODO: Fix this still
    def forward(self, embedded_observation, deterministic):
        x = self.network(torch.cat((embedded_observation, deterministic), 1))
        posterior_dist = create_normal_dist(x, min_std=self.config.min_std)
        posterior = posterior_dist.rsample()
        return posterior_dist, posterior


class RewardModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.parameters.dreamer.reward
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        self.network = build_network(
            self.stochastic_size + self.deterministic_size,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
            1,
        )

    def forward(self, posterior, deterministic):
        x = horizontal_forward(
            self.network, posterior, deterministic, output_shape=(1,)
        )
        dist = create_normal_dist(x, std=1, event_shape=1)
        return dist


class ContinueModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.parameters.dreamer.continue_
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        self.network = build_network(
            self.stochastic_size + self.deterministic_size,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
            1,
        )

    def forward(self, posterior, deterministic):
        x = horizontal_forward(
            self.network, posterior, deterministic, output_shape=(1,)
        )
        dist = torch.distributions.Bernoulli(logits=x)
        return dist