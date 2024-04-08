from algorithms.agent import WorldModel, ImagBehavior
from ruamel.yaml import YAML
from data.pipeline import Pipeline
import torch
from tqdm import tqdm
from modules import tools
import numpy as np
import csv
import os

data_directory = "data/SimulatedData4-8/mixed/valid"
model_path = "models/SimulatedDataModel4-8/model.pt"

def main():
    print("Staring validation...")

    with open("configs.yaml") as f:
        yaml = YAML()
        configs = yaml.load(f)

    obs_space = 23
    act_space = 4
    batch_size = 512
    sequence_length = 32

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    model = WorldModel(obs_space, act_space, configs)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    model.to(configs['device'])

    pipeline = Pipeline(os.path.join(data_directory, "states.csv"), os.path.join(data_directory, "actions.csv"), os.path.join(data_directory, "rewards.csv"))
    dataloader = pipeline.read_csv(sequence_length=sequence_length, batch_size=batch_size)

    history_size = configs['rssm']['history_size']

    error = 0.0

    for batch_count, (states, actions, rewards) in enumerate(tqdm(dataloader, desc=f"Validation")):
        states = states.to(configs['device'])
        actions = actions.to(configs['device'])
        rewards = rewards.to(configs['device'])

        outputs = model._valid({'state': states, 'action': actions, 'reward': rewards})
        error += mean_squared_error(states, outputs)
        print(f"Batch {batch_count} Error: {error}")
        torch.cuda.empty_cache()

    print(f"Mean Squared Error: {error/len(dataloader)}")

    #output = outputs.to('cpu')  # Move the tensor to CPU memory

    #output_numpy = output.squeeze(0).detach().numpy()
    
    #np.savetxt('output.csv', output_numpy, delimiter=',', fmt='%.8f')

def mean_squared_error(actual, predicted):
    return torch.mean((actual - predicted).pow(2))

if __name__ == "__main__":
    main()