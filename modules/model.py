import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from torch.distributions import Normal
import numpy as np

from utils import build_network, horizontal_forward, get_dist


class RSSM(nn.Module):
    def __init__(self, action_size, config):
        super().__init__()
        self.config = config.rssm

        self.recurrent_model = RecurrentModel(action_size, config)
        self.transition_model = TransitionModel(config)
        self.representation_model = RepresentationModel(config)

    def recurrent_model_input_init(self, batch_size):
        return self.transition_model.input_init(
            batch_size
        ), self.recurrent_model.input_init(batch_size)
    
    def img_step(self, prev_state, prev_action):
        """
        img_step computes the prior stochastic state (z-hat) and deterministic hidden state (h_t) 
        based on the prev_state and prev_action

        :param prev_state: previous deterministic & stochastic hidden state (in a dictionary)
        :param prev_action: last action
        :return: prior state dictionary {'stoch': z-hat value, 'deter': h_t value}
        """ 
        x = self.recurrent_model(prev_state, prev_action)
        z_hat_dist, z_hat = self.transition_model(x)
        prior = {'stoch': z_hat_dist, 'deter': z_hat}

        return prior
    
    def obj_step(self, prev_state, prev_action, embed):
        """
        obj_step computes the posterior stochastic state (z) based on the prev_state and prev_action

        :param prev_state: previous deterministic & stochastic hidden state (in a dictionary)
        :param prev_action: last action
        :return: prior state dictionary {'stoch': z-hat value, 'deter': h_t value}
        """ 
        prior = self.img_step(prev_state, prev_action)
        z_dist, z = self.representation_model(prior['deter'], embed)
        post = {'stoch': z_dist, 'deter': z}
        return post, prior

class RecurrentModel(nn.Module):
    def __init__(self, action_size, config):
        super().__init__()
        self.config = config.rssm
        self.device = config.device
        self.stochastic_size = config.rssm.stochastic_size
        self.deterministic_size = config.rssm.deterministic_size

        self.activation = getattr(nn, self.config.act)()

        #TODO: Be able to change the number of layers for linear and gru through config
        self.linear = nn.Linear(
            self.stochastic_size + action_size, self.config.hidden_size
        )
        self.recurrent = nn.GRUCell(self.config.hidden_size, self.deter)

    def forward(self, prev_state, action):
        shape = prev_state["stoch"].shape[:-2] + [self.config.stoch * self.config.classes]
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
        self.config = config.rssm.transition_model
        self.device = config.device
        self.stochastic_size = config.rssm.stoch
        self.deterministic_size = config.rssm.deter

        self.network = build_network(
            self.deter,
            self.config.hidden_size,
            self.config.prior_layers,
            self.config.act,
            self.stochastic_size * 2,
        )

    def forward(self, x):
        x = self.network(x)
        prior_dist = get_dist(self.create_stoch(x, self.config.stoch, self.config.classes, self.config.unimix))
        prior = prior_dist.sample()
        return prior_dist, prior

    def input_init(self, batch_size):
        return torch.zeros(batch_size, self.stochastic_size).to(self.device)

class RepresentationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.rssm
        self.embedded_state_size = config.rssm.embedded_state_size
        self.stochastic_size = config.rssm.stoch
        self.deterministic_size = config.rssm.deter

        self.network = build_network(
            self.embedded_state_size + self.deter,
            self.config.hidden_size,
            self.config.post_layers,
            self.config.act,
            self.stochastic_size * 2,
        )

    def forward(self, embedded_observation, deterministic):
        """
        forward prop through the representation model

        :param embedded_observation: 
        :param prev_action: last action
        :return: prior state dictionary {'stoch': z-hat value, 'deter': h_t value}
        """ 
        x = self.network(torch.cat((embedded_observation, deterministic), 1))
        post_dist = get_dist(self.create_stoch(x, self.config.stoch, self.config.classes, self.config.unimix))
        post = post_dist.sample()
        return post_dist, post


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