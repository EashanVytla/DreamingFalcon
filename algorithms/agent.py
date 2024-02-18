import ruamel.yaml as yaml
import torch
import torch.nn as nn

class Agent(nn.Module):

  configs = yaml.YAML(typ='safe').load('./configs.yaml').read()

  def __init__(self, obs_space, act_space, step):
    self.obs_space = obs_space
    self.act_space = act_space
    self.step = step

class WorldModel(nn.Module):
  def __init__(self):
    self.obs