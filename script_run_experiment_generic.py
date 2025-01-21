
from elliot.run import run_experiment
import warnings
import argparse

# Suppress all warnings
warnings.filterwarnings("ignore")

# read the dataset
parser = argparse.ArgumentParser(description="Run a script to run experiments on a generic dataset.")
parser.add_argument('--type', type=str)
parser.add_argument('--n_hash', type=str)
parser.add_argument('--sampling', type=str)
parser.add_argument('--neighbors', type=int)
args = parser.parse_args()

# take the type of experiment the input arguments
n = args.n_hash

# take the version of the experiment from the input arguments
s = args.sampling

# take the dataset from the input arguments
k = args.neighbors

# take the type of experiment from the input arguments
t = args.type

print(f"Done! We are now starting the Fair ANN Elliot's experiment with YELP dataset")
run_experiment(f"config_run_experiments_yelp/{t}_experiment_yelp_n_hash={n}_sampling={s}_neighbors={k}.yml")