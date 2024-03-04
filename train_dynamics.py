from algorithms.agent import WorldModel
import ruamel.yaml as yaml

def main():
    configs = yaml.YAML(typ='safe').load('configs.yaml').read()
    obs_space = (23,1)
    act_space = (4, 1)
    model = WorldModel(obs_space, act_space, configs)
    model.train()

if __name__ == "__main__":
    main()