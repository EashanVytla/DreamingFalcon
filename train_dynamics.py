from algorithms.agent import WorldModel
import ruamel.yaml as yaml
from data.pipeline import Pipeline

def main():
    configs = yaml.YAML(typ='safe').load('configs.yaml').read()
    obs_space = (23,1)
    act_space = (4, 1)
    model = WorldModel(obs_space, act_space, configs)

    pipeline = Pipeline(r"data/states.csv", r"data/actions.csv")
    pipeline.read_csv()
    pipeline.prepare_data()

    model.train(pipeline.get_data())

if __name__ == "__main__":
    main()