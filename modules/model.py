import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from torch.distributions import Normal
import numpy as np

from utils.utils import build_network, horizontal_forward, uniform_weight_init, ContDist

class RSSM(nn.Module):
    def __init__(self, action_size, config):
        super().__init__()
        self.config = config['rssm']

        self.action_size = action_size

        self.device = config['device']

        self.recurrent_model = RecurrentModel(action_size, config)
        self.transition_model = TransitionModel(config)
        self.representation_model = RepresentationModel(config)

        self.recurrent_model.to(self.device)
        self.transition_model.to(self.device)
        self.representation_model.to(self.device)

    def recurrent_model_input_init(self, batch_size):
        #returns post, actions
        deter = torch.zeros(batch_size, self.config['deter']).to(self.device)
        state = dict(
                mean=torch.zeros([batch_size, self.config['stoch']]).to(self.device),
                std=torch.zeros([batch_size, self.config['stoch']]).to(self.device),
                stoch=torch.zeros([batch_size, self.config['stoch']]).to(self.device),
                deter=deter,
            )
        return state
    
    def img_step(self, prev_state, prev_action):
        """
        img_step computes the prior stochastic state (z-hat) and deterministic hidden state (h_t) 
        based on the prev_state and prev_action

        :param prev_state: previous deterministic & stochastic hidden state (in a dictionary)
        :param prev_action: last action
        :return: prior state dictionary {'stoch': z-hat value, 'deter': h_t value}
        """ 
        x = self.recurrent_model(prev_state, prev_action)
        z_hat_dist, stats = self.transition_model(x)
        prior = {'stoch': z_hat_dist, 'deter': x, 'mean': stats['mean'], 'std': stats['std']}

        return prior
    
    def obs_step(self, prev_state, prev_action, embed):
        """
        obj_step computes the posterior stochastic state (z) based on the prev_state and prev_action

        :param prev_state: previous deterministic & stochastic hidden state (in a dictionary)
        :param prev_action: last action
        :return: prior state dictionary {'stoch': z-hat value, 'deter': h_t value}
        """
        prior = self.img_step(prev_state, prev_action)

        z_dist, stats = self.representation_model(prior['deter'], embed)
        post = {'stoch': z_dist, 'deter': prior['deter'], 'mean': stats['mean'], 'std': stats['std']}
        return post, prior

    def observation(self, embed, action):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))

        prev_state = self.recurrent_model_input_init(len(action))
        #prev_action = torch.zeros((len(action), self.action_size)).to(self.device)

        # Initialize empty tensors to store post and prior
        all_posts = dict(mean=torch.zeros(len(action), action.shape[1], self.config['stoch'], device=self.device),
                         std=torch.zeros(len(action), action.shape[1], self.config['stoch'], device=self.device),
                         )
        all_priors = dict(mean=torch.zeros(len(action), action.shape[1], self.config['stoch'], device=self.device),
                         std=torch.zeros(len(action), action.shape[1], self.config['stoch'], device=self.device),
                         )

        action = swap(action)
        embed = swap(embed)

        prev_action = action[0]

        outputs = []

        for i in range(1, action.size(dim=0)):
            post, prior = self.obs_step(prev_state, prev_action, embed[i-1])
            all_posts['mean'][:, i, :] = post['mean']
            all_priors['mean'][:, i, :] = prior['mean']
            all_posts['std'][:, i, :] = post['std']
            all_priors['std'][:, i, :] = prior['std']
            prev_state = post
            prev_action = action[i]

        return all_posts, all_priors

    def kl_loss(self, post, prior, dyn_scale, rep_scale):
        kld = td.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld(dist(post)._dist, dist(sg(prior))._dist)
        dyn_loss = kld(dist(sg(post))._dist, dist(prior)._dist)

        rep_loss = torch.clip(rep_loss, min=1)
        dyn_loss = torch.clip(dyn_loss, min=1)
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss
    
    def get_dist(self, state, dtype=None):
        #if self._discrete:
        #    logit = state["logit"]
        #    dist = td.independent.Independent(
        #        self.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
        #    )
        #else:
        dist = ContDist(
            td.independent.Independent(td.normal.Normal(state["mean"], state["std"]), 1)
        )
        return dist


class RecurrentModel(nn.Module):
    def __init__(self, action_size, config):
        super().__init__()
        self.config = config['rssm']
        self.device = config['device']
        self.stochastic_size = self.config['stoch']
        self.deterministic_size = self.config['deter']

        self.activation = getattr(nn, self.config['act'])()

        #TODO: Be able to change the number of layers for linear and gru through config
        self.linear = nn.Linear(
            self.stochastic_size + action_size, self.config['hidden_size']
        )
        self.recurrent = nn.GRUCell(self.config['hidden_size'], self.deterministic_size)

    def forward(self, prev_state, action):
        x = torch.cat((prev_state['stoch'], action), dim=1)
        x = self.activation(self.linear(x))
        x = self.recurrent(x, prev_state['deter'])
        return x

    def input_init(self, batch_size):
        return torch.zeros(batch_size, self.deterministic_size).to(self.device)


class TransitionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config['rssm']
        self.device = config['device']
        self.stochastic_size = config['rssm']['stoch']
        self.deterministic_size = config['rssm']['deter']

        self.network = build_network(
            self.deterministic_size,
            self.config['hidden_size'],
            self.config['prior_layers'],
            self.config['act'],
            self.config['hidden_size'],
        )

        self._imgs_stat_layer = nn.Linear(self.config['hidden_size'], 2 * self.config['stoch'])
        self._imgs_stat_layer.apply(uniform_weight_init(1.0))

    def forward(self, x):
        x = self.network(x)
        stats = self._suff_stats_layer(x)
        prior_dist = self.get_dist(stats)
        prior = prior_dist.sample()
        return prior, stats
    
    def _suff_stats_layer(self, x):
        x = self._imgs_stat_layer(x)
        mean, std = torch.split(x, [self.stochastic_size] * 2, -1)
        mean = {
            "none": lambda: mean,
            "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
        }[self.config['dyn_mean_act']]()
        std = {
            "softplus": lambda: F.softplus(std),
            "abs": lambda: torch.abs(std + 1),
            "sigmoid": lambda: torch.sigmoid(std),
            "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
        }[self.config['dyn_std_act']]()
        std = std + self.config['dyn_min_std']
        return {"mean": mean, "std": std}
    
    def get_dist(self, state, dtype=None):
        #if self._discrete:
        #    logit = state["logit"]
        #    dist = td.independent.Independent(
        #        self.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
        #    )
        #else:
        dist = ContDist(
            td.independent.Independent(td.normal.Normal(state["mean"], state["std"]), 1)
        )
        return dist

    def input_init(self, batch_size):
        return torch.zeros(batch_size, self.stochastic_size).to(self.device)

class RepresentationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config['rssm']
        self.embedded_state_size = config['rssm']['embedded_state_size']
        self.stochastic_size = config['rssm']['stoch']
        self.deterministic_size = config['rssm']['deter']

        self.network = build_network(
            self.embedded_state_size + self.deterministic_size,
            self.config['hidden_size'],
            self.config['post_layers'],
            self.config['act'],
            self.config['hidden_size'],
        )

        self._obs_stat_layer = nn.Linear(self.config['hidden_size'], 2 * self.config['stoch'])
        self._obs_stat_layer.apply(uniform_weight_init(1.0))

    def forward(self, embedded_observation, deterministic):
        """
        forward prop through the representation model

        :param embedded_observation: 
        :param prev_action: last action
        :return: prior state dictionary {'stoch': z-hat value, 'deter': h_t value}
        """ 
        x = self.network(torch.cat((embedded_observation, deterministic), 1))
        stats = self._suff_stats_layer(x)
        post_dist = self.get_dist(stats)
        post = post_dist.sample()
        return post, stats
    
    def _suff_stats_layer(self, x):
        x = self._obs_stat_layer(x)
        mean, std = torch.split(x, [self.stochastic_size] * 2, -1)
        mean = {
            "none": lambda: mean,
            "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
        }[self.config['dyn_mean_act']]()
        std = {
            "softplus": lambda: F.softplus(std),
            "abs": lambda: torch.abs(std + 1),
            "sigmoid": lambda: torch.sigmoid(std),
            "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
        }[self.config['dyn_std_act']]()
        std = std + self.config['dyn_min_std']
        return {"mean": mean, "std": std}
    
    def get_dist(self, state, dtype=None):
        #if self._discrete:
        #    logit = state["logit"]
        #    dist = td.independent.Independent(
        #        self.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
        #    )
        #else:
        dist = ContDist(
            td.independent.Independent(td.normal.Normal(state["mean"], state["std"]), 1)
        )
        return dist