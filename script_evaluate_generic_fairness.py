from elliot.run import run_experiment
import warnings
import argparse

# Suppress all warnings
warnings.filterwarnings("ignore")

# read the dataset
parser = argparse.ArgumentParser(description="Run a script to evaluate fairness-oriented experiments on a generic dataset.")
parser.add_argument('--type', type=str)
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

# take the type of experiment the input arguments
t = args.type
# take the dataset from the input arguments
d = args.dataset

print(f"Done! We are now starting the Fair ANN Elliot's experiment with {d} dataset")
run_experiment(f"config_evaluate_fairness_oriented/evaluate_{t}_{d}.yml")