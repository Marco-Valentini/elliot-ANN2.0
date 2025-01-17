from elliot.run import run_experiment

# # start by preprocessing movielens 1M dataset
# print("Done! We are now starting preprocessing of movielens 1M dataset")
# run_experiment("config_preprocess_dataset/preprocess_movielens_1m.yml")

# preprocess last fm 1K dataset
# print("Done! We are now starting preprocessing of last fm 1K dataset")
# run_experiment("config_preprocess_dataset/preprocess_lastfm_1K.yml")

# # preprocess amazon books dataset
# print("Done! We are now starting preprocessing of amazon books dataset")
# run_experiment("config_preprocess_dataset/preprocess_amazon_books.yml")

# preprocess yelp dataset
print("Done! We are now starting preprocessing of YELP dataset")
run_experiment("config_preprocess_dataset/preprocess_yelp.yml")