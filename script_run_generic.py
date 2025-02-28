from elliot.run import run_experiment
import warnings
import argparse

# Suppress all warnings
warnings.filterwarnings("ignore")

# read the dataset
parser = argparse.ArgumentParser(description="Run a script to run fairness-oriented experiments on a generic dataset.")
parser.add_argument('config_path', type=str)
# parser.add_argument('--type', type=str)
# parser.add_argument('--dataset', type=str)
args = parser.parse_args()

# # take the type of experiment the input arguments
# t = args.type
# # take the dataset from the input arguments
# d = args.dataset

# take the config path from the input arguments
config = args.config_path

print(f"Done! We are now starting the Fair ANN Elliot's experiment with {config} configuration")
run_experiment(config)