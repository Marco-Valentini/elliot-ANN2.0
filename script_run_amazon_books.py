
from elliot.run import run_experiment
import warnings
import argparse

# Suppress all warnings
warnings.filterwarnings("ignore")

# read the dataset
parser = argparse.ArgumentParser(description="Run a script to run experiments on Amazon Books dataset.")
parser.add_argument('--type', type=str, default='item', choices=['item', 'user'])
parser.add_argument('--version', type=str, default='1a')
args = parser.parse_args()

# take the type of experiment the input arguments
type = args.type

# take the version of the experiment from the input arguments
version = args.version

print("Done! We are now starting the Fair ANN Elliot's experiment with Amazon Books dataset")
run_experiment(f"config_run_experiments/{type}_experiment_amazon_books_{version}.yml")