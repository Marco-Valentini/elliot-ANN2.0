import pandas as pd
import os
import glob
import re

type = 'user' #'item'
# define the number of cutoffs
cutoffs = [1, 5, 10, 20]

# read the first result
file_path_1 = glob.glob(f"results/yelp_{type}/performance_1/rec_cutoff_1_*.tsv")[0]
df_results_1 = pd.read_csv(file_path_1, sep='\t')

file_path_5 = glob.glob(f"results/yelp_{type}/performance_1/rec_cutoff_5*.tsv")[0]
df_results_5 = pd.read_csv(file_path_5, sep='\t')

file_path_10 = glob.glob(f"results/yelp_{type}/performance_1/rec_cutoff_10*.tsv")[0]
df_results_10 = pd.read_csv(file_path_10, sep='\t')

file_path_20 = glob.glob(f"results/yelp_{type}/performance_1/rec_cutoff_20*.tsv")[0]
df_results_20 = pd.read_csv(file_path_20, sep='\t')

# iterate over the folders
for i in range(2, 46):
    file_path_1 = glob.glob(f"results/yelp_{type}/performance_{i}/rec_cutoff_1_*.tsv")[0]
    df_results_1 = pd.concat([df_results_1, pd.read_csv(file_path_1, sep='\t')], axis=0)

    file_path_5 = glob.glob(f"results/yelp_{type}/performance_{i}/rec_cutoff_5*.tsv")[0]
    df_results_5 = pd.concat([df_results_5, pd.read_csv(file_path_5, sep='\t')], axis=0)

    file_path_10 = glob.glob(f"results/yelp_{type}/performance_{i}/rec_cutoff_10*.tsv")[0]
    df_results_10 = pd.concat([df_results_10, pd.read_csv(file_path_10, sep='\t')], axis=0)

    file_path_20 = glob.glob(f"results/yelp_{type}/performance_{i}/rec_cutoff_20*.tsv")[0]
    df_results_20 = pd.concat([df_results_20, pd.read_csv(file_path_20, sep='\t')], axis=0)

# before saving I want to order the rows based on the name in model,
# first I put all the rows having ItemKNN, then Item Fair ANN , then Item ANN faiss, then Item
# ANNOY and finally ItemKNN fairness

if type == 'item':
    model_order = {
        r'^ItemKNN_nn': 1,
        r'^ItemFairANN': 2,
        r'^ItemANNfaissLSH': 3,
        r'^ItemANNOY': 4,
        r'^ItemKNNfairness': 5
    }
elif type == 'user':
    model_order = {
        r'^UserKNN_nn': 1,
        r'^UserFairANN': 2,
        r'^UserANNfaissLSH': 3,
        r'^UserANNOY': 4,
        r'^UserKNNfairness': 5
    }

# Create a sorting key function using regex
def sort_key(model_name):
    for pattern, order in model_order.items():
        if re.match(pattern, model_name):
            return order
    return float('inf')

# Apply the custom sorting to the dataframes
df_results_1['sort_key'] = df_results_1['model'].apply(sort_key)
df_results_1 = df_results_1.sort_values(by='sort_key').drop(columns='sort_key')

df_results_5['sort_key'] = df_results_5['model'].apply(sort_key)
df_results_5 = df_results_5.sort_values(by='sort_key').drop(columns='sort_key')

df_results_10['sort_key'] = df_results_10['model'].apply(sort_key)
df_results_10 = df_results_10.sort_values(by='sort_key').drop(columns='sort_key')

df_results_20['sort_key'] = df_results_20['model'].apply(sort_key)
df_results_20 = df_results_20.sort_values(by='sort_key').drop(columns='sort_key')

# save the results
df_results_1.to_csv(f"results/yelp_{type}/performance/rec_cutoff_1.tsv", sep='\t', index=False)
df_results_5.to_csv(f"results/yelp_{type}/performance/rec_cutoff_5.tsv", sep='\t', index=False)
df_results_10.to_csv(f"results/yelp_{type}/performance/rec_cutoff_10.tsv", sep='\t', index=False)
df_results_20.to_csv(f"results/yelp_{type}/performance/rec_cutoff_20.tsv", sep='\t', index=False)
