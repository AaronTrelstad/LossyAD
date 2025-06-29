from src.configs import ExperimentConfig, set_seed
from src.experiment import run_experiment

if __name__ == '__main__':
    config = ExperimentConfig()
    set_seed(config.seed)

    print("Initalized")

    run_experiment(config)
