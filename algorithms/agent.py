import torch
import torch.nn as nn
from modules import encoder, decoder, model
from utils import build_network, RequiresGrad

class WorldModel(nn.Module):
  def __init__(self, obs_space, act_space, config):
    super(WorldModel, self).__init__()
    self._use_amp = True if config.precision == 16 else False

    self._config = config

    input_size = 23

    self.encoder = build_network(
      input_size,
      config.encoder.mlp_units,
      config.encoder.mlp_layers,
      config.decoder.elu,
      config.rssm.embedded_state_size
    )

    self.rssm = model.RSSM(act_space,config)

    self.heads = nn.ModuleDict()
    self.heads['decoder'] = build_network(
      config.rssm.deter + config.rssm.stoch,
      config.decoder.mlp_units,
      config.decoder.mlp_layers,
      config.decoder.elu,
      input_size
    )

    self.params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.rssm.parameters())
            + list(self.heads['decoder'].parameters())
        )

    optimizer = torch.optim.Adam(self.model_params, lr=0.0001)

  def train(self, data):
    with RequiresGrad(self):
      with torch.cuda.amp.autocast(self._use_amp):
        embed = self.encoder(data['state'])
        post, prior = self.rssm.observation(embed, data['action'])

        kl_scale = self.config.loss_scales.kl
        dyn_scale = self.config.loss_scales.dyn
        rep_scale = self.config.loss_scales.rep

        kl_loss, kl_value, dyn_loss, rep_loss = self.rssm.kl_loss(
                    post, prior, kl_scale, dyn_scale, rep_scale
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