import torch
import torch.nn as nn
from modules import model
from utils.utils import build_network, RequiresGrad, uniform_weight_init

class WorldModel(nn.Module):
  def __init__(self, obs_space, act_space, config):
    super(WorldModel, self).__init__()
    self._use_amp = True if config['precision'] == 16 else False
    self.config = config

    self.encoder = build_network(
      obs_space,
      config['encoder']['mlp_units'],
      config['encoder']['mlp_layers'],
      config['decoder']['act'],
      config['rssm']['embedded_state_size']
    )

    self.rssm = model.RSSM(act_space,config)

    self.heads = nn.ModuleDict()
    self.heads['decoder'] = build_network(
      config['rssm']['deter'] + config['rssm']['stoch'],
      config['decoder']['mlp_units'],
      config['decoder']['mlp_layers'],
      config['decoder']['act'],
      obs_space
    )

    self.encoder.to(self.config['device'])
    self.heads['decoder'].to(self.config['device'])

    self.params = (
            list(self.encoder.parameters())
            + list(self.heads['decoder'].parameters())
            + list(self.rssm.parameters())
            + list(self.heads['decoder'].parameters())
        )

    optimizer = torch.optim.Adam(self.params, lr=0.0001)

  def train(self, data):
    with RequiresGrad(self):
      with torch.cuda.amp.autocast(self._use_amp):
        data['state'] = data['state'].float()
        data['action'] = data['action'].float()
        embed = self.encoder(data['state'])
        post, prior = self.rssm.observation(embed, data['action'])

        dyn_scale = self.config['loss_scales']['dyn']
        rep_scale = self.config['loss_scales']['rep']

        kl_loss, kl_value, dyn_loss, rep_loss = self.rssm.kl_loss(
                    post, prior, dyn_scale, rep_scale
                )
        
        assert kl_loss.shape == embed.shape[:2], kl_loss.shape

        preds = {}
        for name, head in self.heads.items():
          grad_head = name in self._config.grad_heads
          feat = torch.cat([post['stoch'], post['deter']], -1)
          pred = head(feat)
          if type(pred) is dict:
            preds.update(pred)
          else:
            preds[name] = pred

        model_loss = -preds['decoder'].log_prob(data['state']) + kl_loss
      metrics = self._model_opt(torch.mean(model_loss), self.parameters())
      
      self.model_optimizer.zero_grad()
      model_loss.backward()
      self.model_optimizer.step()