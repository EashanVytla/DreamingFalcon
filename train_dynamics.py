from algorithms.agent import WorldModel
from ruamel.yaml import YAML
from data.pipeline import Pipeline
import torch
from tqdm import tqdm
from modules import tools
import numpy as np
import os
import sys

obs_space = 11
act_space = 4
num_epochs = 128
sequence_length = 64
batch_size = 350
checkpoint = 50
model_directory = "models/SimulatedDataModel4-14-4"
data_directory_gl = "data/SimulatedData4-13/solo/train"
log_directory = "logs/4-14-4"

def main():
    if len(sys.argv) > 2:
        print(f"Loading data from {sys.argv[2]}")
        data_directory = sys.argv[2]
        step = num_epochs
    else:
        data_directory = data_directory_gl
        step = 0
    assert os.path.exists(data_directory)

    with open("configs.yaml") as f:
        yaml = YAML()
        configs = yaml.load(f)

    logger = tools.Logger(log_directory, step)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    model = WorldModel(obs_space, act_space, configs)

    if len(sys.argv) > 1:
        model.requires_grad_(requires_grad=False)
        print(f"Loading model from {sys.argv[1]}")
        model.load_state_dict(torch.load(sys.argv[1]))
        model.requires_grad_(requires_grad=True)

    model.to(configs['device'])

    pipeline = Pipeline(os.path.join(data_directory, "states.csv"), os.path.join(data_directory, "actions.csv"), os.path.join(data_directory, "rewards.csv"))
    dataloader = pipeline.read_csv(sequence_length=sequence_length, batch_size=batch_size)

    # Check if the directory exists
    if not os.path.exists(model_directory):
        # If not, create the directory
        os.makedirs(model_directory)

    for epoch_count in range(num_epochs):
        for batch_count, (states, actions, rewards) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch_count}")):
        #for batch_count, (states, actions) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch_count}")):
            states = states.to(configs['device'])
            actions = actions.to(configs['device'])
            rewards = rewards.to(configs['device'])

            post, context, metrics = model._train({'state': states, 'action': actions, 'reward': rewards})
            #post, context, metrics = model._train({'state': states, 'action': actions})

            for name, values in metrics.items():
                logger.scalar(name, float(np.mean(values)))
                metrics[name] = []
        logger.write(step=epoch_count + step)

        if epoch_count % checkpoint == 0 and epoch_count != 0:
            torch.save(model.state_dict(), os.path.join(model_directory, "checkpoint.pt"))
            print("Checkpoint Model saved!")

    torch.save(model.state_dict(), os.path.join(model_directory, "model.pt"))

    print("Model saved!")

if __name__ == "__main__":
    main()