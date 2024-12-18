
from elliot.run import run_experiment

print("Done! We are now starting the Evaluation of Fair ANN Elliot's experiment with Item-based methods")
run_experiment("config_files/compute_metrics/evaluate_item_movielens_100k.yml")

# print("Done! We are now starting the Evaluation of Fair ANN Elliot's experiment with User-based methods")
# run_experiment("config_files/compute_metrics/evaluate_user_movielens_100k.yml")