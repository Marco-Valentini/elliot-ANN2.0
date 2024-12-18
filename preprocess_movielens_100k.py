from elliot.run import run_experiment

print("We are now starting the data preprocessing with Elliot")
run_experiment("config_preprocess_dataset/preprocess_movielens_100k.yml")