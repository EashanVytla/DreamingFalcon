from algorithms.agent import WorldModel
from ruamel.yaml import YAML
from data.pipeline import Pipeline
import torch
from tqdm import tqdm

def main():
    with open("configs.yaml") as f:
        yaml = YAML()
        configs = yaml.load(f)

    obs_space = 17
    act_space = 4

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    model = WorldModel(obs_space, act_space, configs)

    model.to(configs['device'])

    pipeline = Pipeline("data/2023-10-13-07-28-08/states.csv", "data/2023-10-13-07-28-08/actions.csv")
    dataloader = pipeline.read_csv()

    for batch_count, (states, actions) in enumerate(tqdm(dataloader, desc="Epoch 1")):
        states = states.to('cuda')
        actions = actions.to('cuda')
        model.train({'state': states, 'action': actions})
        if(batch_count == 1):
            break

if __name__ == "__main__":
    main()