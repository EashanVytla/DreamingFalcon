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
        #returns post, prior
        return self.representation_model.input_init(batch_size), self.transition_model.input_init(batch_size)
    
    def img_step(self, prev_state, prev_action):
        """
        img_step computes the prior stochastic state (z-hat) and deterministic hidden state (h_t) 
        based on the prev_state and prev_action

        :param prev_state: previous deterministic & stochastic hidden state (in a dictionary)
        :param prev_action: last action
        :return: prior state dictionary {'stoch': z-hat value, 'deter': h_t value}
        """ 
        x = self.recurrent_model(prev_state, prev_action)
        z_hat_dist = self.transition_model(x)
        prior = {'stoch': z_hat_dist, 'deter': x}

        return prior
    
    def obs_step(self, prev_state, prev_action, embed):
        """
        obj_step computes the posterior stochastic state (z) based on the prev_state and prev_action

        :param prev_state: previous deterministic & stochastic hidden state (in a dictionary)
        :param prev_action: last action
        :return: prior state dictionary {'stoch': z-hat value, 'deter': h_t value}
        """
        prior = self.img_step(prev_state, prev_action)
        z_dist = self.representation_model(prior['deter'], embed)
        post = {'stoch': z_dist, 'deter': prior['deter']}
        return post, prior

    def observation(self, embed, action):
        prev_state, prior = self.recurrent_model_init(len(action))
        prev_action = action[0]
        outputs = []

        for i in range(1, len(action)):
            outputs.append = self.obs_step(prev_state, prev_action, embed)
            prev_state = outputs[i][0]
            prev_action = action[i]

        return outputs

    
    def kl_loss(self, post, prior, dyn_scale, rep_scale):
        kld = td.kl.kl_divergence
        dist = lambda x: get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld(dist(post)._dist, dist(sg(prior))._dist,)
        dyn_loss = kld(dist(sg(post))._dist, dist(prior)._dist)

        rep_loss = torch.clip(rep_loss, min=1)
        dyn_loss = torch.clip(dyn_loss, min=1)
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss


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
        return prior

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
        return post