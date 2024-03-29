from algorithms.agent2 import WorldModel
from ruamel.yaml import YAML
from data.pipeline import Pipeline
import torch
from tqdm import tqdm
from testModel import tools
import numpy as np

def main():
    with open("configs.yaml") as f:
        yaml = YAML()
        configs = yaml.load(f)

    obs_space = 9
    act_space = 4
    num_epochs = 200

    logger = tools.Logger("logs/3-29", 0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    model = WorldModel(obs_space, act_space, configs)

    model.to(configs['device'])

    pipeline = Pipeline("data/SimulatedData1/states.csv", "data/SimulatedData1/actions.csv")
    dataloader = pipeline.read_csv()

    for epoch_count in range(num_epochs):
        for batch_count, (states, actions) in enumerate(tqdm(dataloader, desc="Epoch 1")):
            states = states.to('cuda')
            actions = actions.to('cuda')
            post, context, metrics = model._train({'state': states, 'action': actions})
            for name, values in metrics.items():
                logger.scalar(name, float(np.mean(values)))
                metrics[name] = []
        logger.write(step=epoch_count)
    
    torch.save(model.state_dict(), r"models/model.pt")

if __name__ == "__main__":
    main()