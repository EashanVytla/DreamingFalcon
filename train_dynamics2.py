from algorithms.agent2 import WorldModel
from ruamel.yaml import YAML
from data.pipeline import Pipeline
import torch
from tqdm import tqdm
from testModel import tools
import numpy as np
import os

obs_space = 9
act_space = 4
num_epochs = 100
sequence_length = 50
batch_size = 50 #Change to 500
model_directory = "models/SimulatedDataModel1"
data_directory = "data/SimulatedData1"
log_directory = "logs/4-1"

def main():
    assert os.path.exists(data_directory)

    with open("configs.yaml") as f:
        yaml = YAML()
        configs = yaml.load(f)

    logger = tools.Logger(log_directory, 0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    model = WorldModel(obs_space, act_space, configs)

    model.to(configs['device'])

    pipeline = Pipeline(os.path.join(data_directory, "states.csv"), os.path.join(data_directory, "actions.csv"))
    dataloader = pipeline.read_csv(sequence_length=sequence_length, batch_size=batch_size)

    # Check if the directory exists
    if not os.path.exists(model_directory):
        # If not, create the directory
        os.makedirs(model_directory)

    for epoch_count in range(num_epochs):
        for batch_count, (states, actions) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch_count}")):
            states = states.to('cuda')
            actions = actions.to('cuda')
            post, context, metrics = model._train({'state': states, 'action': actions})
            for name, values in metrics.items():
                logger.scalar(name, float(np.mean(values)))
                metrics[name] = []
        logger.write(step=epoch_count)

        if epoch_count % 2 == 0 and epoch_count != 0:
            torch.save(model.state_dict(), os.path.join(model_directory, "checkpoint.pt"))

    torch.save(model.state_dict(), os.path.join(model_directory, "model.pt"))

if __name__ == "__main__":
    main()