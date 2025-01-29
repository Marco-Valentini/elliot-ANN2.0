import argparse
from elliot.run import run_experiment
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# read the dataset
parser = argparse.ArgumentParser(description="Run a script to evaluate experiments on a generic dataset.")
parser.add_argument('--type', type=str)
parser.add_argument('--i', type=str)
args = parser.parse_args()

# take the type of experiment the input arguments
i = args.i

# take the type of experiment from the input arguments
t = args.type

print(f"Done! We are now starting the Fair ANN Elliot's experiment with YELP dataset")
run_experiment(f"config_compute_metrics/evaluate_{t}_yelp_{i}.yml")