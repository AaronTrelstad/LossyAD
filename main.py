from src.configs import ExperimentConfig, set_seed
from src.experiment import run_experiment
from src.analysis import run_analysis

if __name__ == '__main__':
    config = ExperimentConfig()
    set_seed(config.seed)

    run_experiment(config)

    run_analysis(config)

