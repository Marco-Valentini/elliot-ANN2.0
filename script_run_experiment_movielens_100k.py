
from elliot.run import run_experiment
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
#
# print("Done! We are now starting the Fair ANN Elliot's experiment with Item-based methods")
# run_experiment("config_run_experiments/item_experiment_movielens_100k.yml")

print("Done! We are now starting the Fair ANN Elliot's experiment with User-based methods")
run_experiment("config_run_experiments/user_experiment_movielens_100k.yml")