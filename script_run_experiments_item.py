
from elliot.run import run_experiment
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

print("Done! We are now starting the Fair ANN Elliot's experiment with Item-based methods")

# run_experiment("config_run_experiments/item_experiment_movielens_1m.yml")

# run_experiment("config_run_experiments/item_experiment_lastfm_1k.yml")

# run_experiment("config_run_experiments/item_experiment_amazon_books.yml")

run_experiment("config_run_experiments/item_experiment_yelp.yml")