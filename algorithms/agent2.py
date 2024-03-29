import torch
import torch.nn as nn
from testModel import networks
from testModel import tools
from utils.utils import build_network
import copy

to_np = lambda x: x.detach().cpu().numpy()

class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()

class WorldModel(nn.Module):
  def __init__(self, obs_space, act_space, config):
    super(WorldModel, self).__init__()
    self._use_amp = True if config['precision'] == 16 else False
    self.config = config

    self.act_space = act_space

    self.history_size = config['rssm']['history_size']

    self.input_size = obs_space * self.history_size + act_space * self.history_size

    self.encoder = networks.MultiEncoder(
        input_size=self.input_size,
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
      mlp_shapes=(1, self.input_size),
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
  
  def prepare_data(self, data):
    concatenated_states = torch.zeros((data['state'].shape[0] - self.history_size, data['state'].shape[1], data['state'].shape[2] * self.history_size + data['action'].shape[2] * self.history_size))

    # Loop through the batch dimension
    for batch in range(data['state'].shape[1]):
      # Loop through the batch length dimension
      for i in range(self.history_size, data['state'].shape[0]):
        past_states = data['state'][i-self.history_size:i, batch, :]  # Slicing to get the past 10 states
        past_actions = data['action'][i-self.history_size:i, batch, :]  # Slicing to get the past 10 actions
        
        # Concatenate the past states and actions along the feature dimension
        concatenated_state = torch.cat((past_states.flatten(), past_actions.flatten()), dim=0)
        
        # Store the concatenated state in the output tensor
        concatenated_states[i-self.history_size, batch, :] = concatenated_state

    return concatenated_states

  def _train(self, data):
    with tools.RequiresGrad(self):
      with torch.cuda.amp.autocast(self._use_amp):
        data['state'] = data['state'].float()
        data['action'] = data['action'].float()

        history_states = self.prepare_data(data)

        history_states = history_states.to(device=self.config['device'])

        data['action'] = data['action'][self.history_size:,:,:]

        embed = self.encoder(history_states)

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
          loss = -pred.log_prob(history_states)
          assert loss.shape == embed.shape[:2], (name, loss.shape)
          losses[name] = loss
        scaled = {
          key: value * self._scales.get(key, 1.0)
          for key, value in losses.items()
        }

        model_loss = sum(scaled.values()) + kl_loss
      metrics = self._model_opt(torch.mean(model_loss), self.parameters())

    metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})

    #metrics["kl_free"] = kl_free
    #metrics["dyn_scale"] = dyn_scale
    #metrics["rep_scale"] = rep_scale
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
  
class ImagBehavior(nn.Module):
  def __init__(self, config, world_model):
    super(ImagBehavior, self).__init__()
    self._use_amp = True if config['precision'] == 16 else False
    self._config = config
    self._world_model = world_model
    feat_size = config['rssm']['stoch'] + config['rssm']['deter']
    self.actor = networks.MLP(
      feat_size,
      (world_model.act_space,),
      config['actor']["layers"],
      config['rssm']['units'],
      config['rssm']['act'],
      config['rssm']['norm'],
      config['actor']["dist"],
      config['actor']["std"],
      config['actor']["min_std"],
      config['actor']["max_std"],
      absmax=1.0,
      temp=config['actor']["temp"],
      unimix_ratio=config['actor']["unimix_ratio"],
      outscale=config['actor']["outscale"],
      name="Actor",
    )
    self.value = networks.MLP(
      feat_size,
      (255,) if config['critic']["dist"] == "symlog_disc" else (),
      config['critic']["layers"],
      config['rssm']['units'],
      config['rssm']['act'],
      config['rssm']['norm'],
      config['critic']["dist"],
      outscale=config['critic']["outscale"],
      device=config['device'],
      name="Value",
    )
    if config['critic']["slow_target"]:
      self._slow_value = copy.deepcopy(self.value)
      self._updates = 0
    kw = dict(wd=config['weight_decay'], opt=config['model_opt']['opt'], use_amp=self._use_amp)
    self._actor_opt = tools.Optimizer(
      "actor",
      self.actor.parameters(),
      config['actor']["lr"],
      config['actor']["eps"],
      config['actor']["grad_clip"],
      **kw,
    )
    print(
      f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
    )
    self._value_opt = tools.Optimizer(
      "value",
      self.value.parameters(),
      config['critic']["lr"],
      config['critic']["eps"],
      config['critic']["grad_clip"],
      **kw,
    )
    print(
      f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
    )
    if self._config['reward_EMA']:
      # register ema_vals to nn.Module for enabling torch.save and torch.load
      self.register_buffer("ema_vals", torch.zeros((2,)).to(self._config['device']))
      self.reward_ema = RewardEMA(device=self._config['device'])

  def imagine(self, start, policy, horizon):
    dynamics = self._world_model.rssm
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in start.items()}

    def step(prev, _):
      state, _, _ = prev
      feat = dynamics.get_feat(state)
      inp = feat.detach()
      action = policy(inp).sample()
      succ = dynamics.img_step(state, action)
      return succ, feat, action

    succ, feats, actions = tools.static_scan(
      step, [torch.arange(horizon)], (start, None, None)
    )
    states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

    return feats, states, actions