
from elliot.run import run_experiment
import warnings
import argparse

# Suppress all warnings
warnings.filterwarnings("ignore")

# read the dataset
parser = argparse.ArgumentParser(description="Run a script to run experiments on a generic dataset.")
parser.add_argument('--type', type=str, default='item', choices=['item', 'user'])
parser.add_argument('--version', type=str, default='1a')
parser.add_argument('--dataset', type=str, default='amazon_books', choices=['amazon_books', 'lastfm_1k', 'yelp', 'movielens_100k', 'movielens_1m'])
args = parser.parse_args()

# take the type of experiment the input arguments
type = args.type

# take the version of the experiment from the input arguments
version = args.version

# take the dataset from the input arguments
dataset = args.dataset

print(f"Done! We are now starting the Fair ANN Elliot's experiment with {dataset} dataset")
run_experiment(f"config_run_experiments/{type}_experiment_{dataset}_{version}.yml")