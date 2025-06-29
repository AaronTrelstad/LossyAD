from src.configs import AnalysisConfig
from src.analysis import run_analysis

if __name__ == '__main__':
    config = AnalysisConfig()

    run_analysis(config)


