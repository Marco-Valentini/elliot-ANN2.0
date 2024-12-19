
from elliot.run import run_experiment

print("Done! We are now starting the Fair ANN Elliot's experiment with Item-based methods")

run_experiment("config_run_experiments/item_experiment_movielens_1m.yml")

run_experiment("config_run_experiments/item_experiment_lastfm_1k.yml")

run_experiment("config_run_experiments/item_experiment_amazon_books.yml")

print("Done! We are now starting the Fair ANN Elliot's experiment with User-based methods")

run_experiment("config_run_experiments/user_experiment_movielens_1m.yml")

run_experiment("config_run_experiments/user_experiment_lastfm_1k.yml")

run_experiment("config_run_experiments/user_experiment_amazon_books.yml")