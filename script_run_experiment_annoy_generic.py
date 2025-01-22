
from elliot.run import run_experiment
import warnings
import argparse

# Suppress all warnings
warnings.filterwarnings("ignore")

# read the dataset
parser = argparse.ArgumentParser(description="Run a script to run experiments on a generic dataset with ANNOY-based models.")
parser.add_argument('--type', type=str)
parser.add_argument('--neighbors', type=int)
parser.add_argument('--n_trees', type=int)
parser.add_argument('--search_k', type=int)
args = parser.parse_args()

# take the number of neighbors from the input arguments
k = args.neighbors

# take the version of the experiment from the input arguments
t = args.n_trees

# take the search_k from the input arguments
s = args.search_k

# take the type of experiment from the input arguments
type = args.type

print(f"Done! We are now starting the Fair ANN Elliot's experiment with YELP dataset")
run_experiment(f"config_run_experiments_yelp/{type}_ANNOY_experiment_yelp_neighbors={k}_n_trees={t}_search_k={s}.yml")