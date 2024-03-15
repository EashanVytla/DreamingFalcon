import torch
import torch.nn as nn
from testModel import networks
from testModel import tools
from utils.utils import build_network

to_np = lambda x: x.detach().cpu().numpy()

class WorldModel(nn.Module):
  def __init__(self, obs_space, act_space, config):
    super(WorldModel, self).__init__()
    self._use_amp = True if config['precision'] == 16 else False
    self.config = config

    self.encoder = networks.MultiEncoder(
        input_size=obs_space,
        mlp_keys=self.config['encoder']['mlp_keys'],
        act=self.config['encoder']['act'],
        norm=self.config['encoder']['norm'],
        mlp_layers=self.config['encoder']['mlp_layers'],
        mlp_units=self.config['encoder']['mlp_units'],
        symlog_inputs=self.config['encoder']['symlog_inputs'],
      )

    self.rssm = networks.RSSM(embed=config['rssm']['embedded_state_size'], 
                              num_actions=act_space, 
                              device=config['device'],
                              stoch=config['rssm']['stoch'],
                              deter=config['rssm']['deter'],
                              hidden=config['rssm']['hidden_size'],
                              act=config['rssm']['act'],
                              mean_act=config['rssm']['dyn_mean_act'],
                              std_act=config['rssm']['dyn_std_act'],
                              min_std=config['rssm']['dyn_min_std'],
                              unimix_ratio=config['rssm']['unimix'],
                              initial=config['rssm']['initial']
                              )

    self.heads = nn.ModuleDict()

    self.feat_size = config['rssm']['stoch'] + config['rssm']['deter']
    
    self.heads["decoder"] = networks.MultiDecoder(
      feat_size=self.feat_size,
      mlp_shapes=(1, obs_space),
      mlp_keys=self.config['decoder']['mlp_keys'],
      act=self.config['decoder']['act'],
      norm=self.config['decoder']['norm'],
      mlp_layers=self.config['decoder']['mlp_layers'],
      mlp_units=self.config['decoder']['mlp_units'],
      vector_dist=self.config['decoder']['vector_dist'],
      outscale=self.config['decoder']['outscale'],
    )

    self.encoder.to(self.config['device'])
    self.heads['decoder'].to(self.config['device'])

    self._scales = dict(
            reward=config['reward_head']["loss_scale"],
            cont=config['cont_head']["loss_scale"],
        )
    
    self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config['model_opt']['lr'],
            config['model_opt']['eps'],
            config['model_opt']['clip'],
            config['model_opt']['wd'],
            opt=config['model_opt']['opt'],
            use_amp=self._use_amp,
        )
    
    print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )

  def train(self, data):
    with tools.RequiresGrad(self):
      with torch.cuda.amp.autocast(self._use_amp):
        data['state'] = data['state'].float()
        data['action'] = data['action'].float()

        embed = self.encoder(data['state'])

        post, prior = self.rssm.observe(
          embed, data["action"]
        )

        kl_free = self.config['loss_scales']['kl']
        dyn_scale = self.config['loss_scales']['dyn']
        rep_scale = self.config['loss_scales']['rep']

        kl_loss, kl_value, dyn_loss, rep_loss = self.rssm.kl_loss(
          post, prior, kl_free, dyn_scale, rep_scale
        )

        assert kl_loss.shape == embed.shape[:2], kl_loss.shape

        preds = {}
        for name, head in self.heads.items():
          grad_head = name in self.config['grad_heads']
          feat = self.rssm.get_feat(post)
          feat = feat if grad_head else feat.detach()
          pred = head(feat)
          if type(pred) is dict:
            preds.update(pred)
          else:
            preds[name] = pred

        losses = {}

        for name, pred in preds.items():
          loss = -pred.log_prob(data['state'])
          assert loss.shape == embed.shape[:2], (name, loss.shape)
          losses[name] = loss
        scaled = {
          key: value * self._scales.get(key, 1.0)
          for key, value in losses.items()
        }

        model_loss = sum(scaled.values()) + kl_loss
      metrics = self._model_opt(torch.mean(model_loss), self.parameters())

    metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})

    metrics["kl_free"] = kl_free
    metrics["dyn_scale"] = dyn_scale
    metrics["rep_scale"] = rep_scale
    metrics["dyn_loss"] = to_np(dyn_loss)
    metrics["rep_loss"] = to_np(rep_loss)
    metrics["kl"] = to_np(torch.mean(kl_value))

    with torch.cuda.amp.autocast(self._use_amp):
      metrics["prior_ent"] = to_np(
        torch.mean(self.rssm.get_dist(prior).entropy())
      )
      metrics["post_ent"] = to_np(
        torch.mean(self.rssm.get_dist(post).entropy())
      )
      context = dict(
        embed=embed,
        feat=self.rssm.get_feat(post),
        kl=kl_value,
        postent=self.rssm.get_dist(post).entropy(),
      )
    post = {k: v.detach() for k, v in post.items()}

    return post, context, metrics