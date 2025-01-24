
from elliot.run import run_experiment
import warnings
import argparse

# Suppress all warnings
warnings.filterwarnings("ignore")

# read the dataset
parser = argparse.ArgumentParser(description="Run a script to run experiments on a generic dataset.")
parser.add_argument('--type', type=str)
parser.add_argument('--similarity', type=str)
parser.add_argument('--sampling', type=str)
parser.add_argument('--n_hash', type=int)
parser.add_argument('--n_tables', type=int)
args = parser.parse_args()

# take the type of experiment from the input arguments
type = args.type

# take the similarity from the input arguments
sim = args.similarity

# take the version of the experiment from the input arguments
s = args.sampling

# take the number of hash functions the input arguments
n = args.n_hash

# take the tables from the input arguments
n_tables = args.n_tables



print(f"Done! We are now starting the Fair ANN Elliot's experiment with YELP dataset")
run_experiment(f"config_run_experiments_yelp/{type}_experiment_yelp_sim={sim}_samp_strat={s}_n_hash={n}_n_tables={n_tables}.yml")