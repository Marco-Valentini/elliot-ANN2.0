from elliot.run import run_experiment

# start from movielens 100k dataset
print("Done! We are now starting experiment on movielens 1M dataset")
run_experiment("config_run_experiments/FairANN_configuration_movielens_100k.yml")