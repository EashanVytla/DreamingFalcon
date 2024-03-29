from algorithms.agent2 import WorldModel, ImagBehavior
from ruamel.yaml import YAML
from data.pipeline import Pipeline
import torch
from tqdm import tqdm
from testModel import tools
import numpy as np
import csv

def main():
    with open("configs.yaml") as f:
        yaml = YAML()
        configs = yaml.load(f)

    obs_space = 17
    act_space = 4
    num_iter = 1
    batch_size = 1


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    model = WorldModel(obs_space, act_space, configs)

    model.load_state_dict(torch.load("models/model.pt"))
    model.eval()

    #img_model = ImagBehavior(configs, model)

    model.to(configs['device'])

    pipeline = Pipeline("data/2023-10-13-07-28-08/states.csv", "data/2023-10-13-07-28-08/actions.csv")
    dataloader = pipeline.read_csv(batch_size=batch_size)

    index, (states, actions) = next(enumerate(iter(dataloader), start=3050))

    #prev_state = model.rssm.initial(batch_size)
    states = states.to('cuda')
    actions = actions.to('cuda')
    
    states = states.float()
    actions = actions.float()

    embed = model.encoder(states)

    #post, prior = model.rssm.obs_step(prev_state, actions[:, 0, :], embed_start)
    #actions = actions[:, 1:, :]
    #feats, states, actions = img_model.imagine(post, actions, actions.shape[1])
    
    states, _ = model.rssm.observe(embed, actions)

    init = {k: v[:, 1] for k, v in states.items()}
    
    prior = model.rssm.imagine_with_action(actions, init)

    output = model.heads['decoder'](model.rssm.get_feat(prior)).mode()

    output = output.to('cpu')  # Move the tensor to CPU memory

    output_numpy = output.squeeze(0).detach().numpy()
    
    np.savetxt('output.csv', output_numpy, delimiter=',', fmt='%.8f')

    print(f"Starting Index: {index}")


if __name__ == "__main__":
    main()