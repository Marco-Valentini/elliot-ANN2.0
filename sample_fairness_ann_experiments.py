from elliot.run import run_experiment

# start by preprocessing movielens 1M dataset
print("Done! We are now starting experiment on movielens 1M dataset")
run_experiment("config_run_experiments/FairANN_configuration_movielens_1m.yml")

# preprocess last fm 1K dataset
print("Done! We are now starting experiment on last fm 1K dataset")
run_experiment("config_run_experiments/FairANN_configuration_lastfm_1K.yml")

# preprocess amazon books dataset
print("Done! We are now starting experiment on amazon books dataset")
run_experiment("config_run_experiments/FairANN_configuration_amazon_books.yml")