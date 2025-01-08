
from elliot.run import run_experiment

# print("Done! We are now starting the Evaluation of Fair ANN Elliot's experiment with Item-based methods")
#
# run_experiment("config_compute_metrics/evaluate_item_movielens_1m.yml")
#
# run_experiment("config_compute_metrics/evaluate_item_lastfm_1k.yml")
#
# run_experiment("config_compute_metrics/evaluate_item_amazon_books.yml")

print("Done! We are now starting the Evaluation of Fair ANN Elliot's experiment with User-based methods")

# run_experiment("config_compute_metrics/evaluate_user_movielens_1m.yml")

run_experiment("config_compute_metrics/evaluate_user_lastfm_1k.yml")

# run_experiment("config_compute_metrics/evaluate_user_amazon_books.yml")