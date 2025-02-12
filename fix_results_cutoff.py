import pandas as pd
import glob


# define the dataset
dataset = ['movielens_1m', 'lastfm_1k', 'yelp']

# define the type of experiment
experiment = ['item', 'user']

# define the cutoffs
cutoffs = [1, 5, 10, 20]

paths = []
new_paths = []

for data in dataset:
    for exp in experiment:
        for cutoff in cutoffs:
            # match a regex pattern
            paths.append(glob.glob(f'./results/{data}_{exp}/performance/rec_cutoff_{cutoff}_*.tsv')[0])
            new_paths.append(glob.glob(f'./results/{data}_{exp}/performance/rec_cutoff_{cutoff}_*.tsv')[0].replace('cutoff', 'cutoff_fixed'))

for path, new_path in zip(paths, new_paths):
    df_results = pd.read_csv(path, sep='\t')
    # iterate over the rows of the dataframe
    for i, row in df_results.iterrows():
        # take the first column
        model_row = row['model']
        # extract the first word before '_'
        model_name = model_row.split('_')[0]
        # we need to change something only when the model_name is 'ItemFairANN' or 'ItemANNOY' or 'UserFairANN' or 'UserANNOY'
        if model_name == 'ItemFairANN' or model_name == 'UserFairANN':
            # substitute the expression "samp_strat" with "samp-strat"
            model_row = model_row.replace('samp_strat', 'samp-strat')
            # check if the expression "no_sampling", "weighted_uniform", "approx_degree" is present in the model name
            if 'no_sampling' in model_row:
                # substitute the expression "no_sampling" with "no-sampling"
                model_row = model_row.replace('no_sampling', 'no-sampling')
            elif 'weighted_uniform' in model_row:
                # substitute the expression "weighted_uniform" with "weighted-uniform"
                model_row = model_row.replace('weighted_uniform', 'weighted-uniform')
            elif 'approx_degree' in model_row:
                # substitute the expression "approx_degree" with "approx-degree"
                model_row = model_row.replace('approx_degree', 'approx-degree')
            # substitute the expression "n_h" with "n-h"
            model_row = model_row.replace('n_h', 'n-h')
            # substitute the expression "n_t" with "n-t"
            model_row = model_row.replace('n_t', 'n-t')
            # substitute the expression "sim_thres" with "sim-thres"
            model_row = model_row.replace('sim_thres', 'sim-thres')
            # update the model name in the dataframe
            df_results.loc[i, 'model'] = model_row

        elif model_name == 'ItemANNOY' or model_name == 'UserANNOY':
            # substitute the expression "n_trees" with "n-trees"
            model_row = model_row.replace('n_tr', 'n-tr')
            # substitute the expression "search_k" with "search-k"
            model_row = model_row.replace('s_k', 's-k')
            # update the model name in the dataframe
            df_results.loc[i, 'model'] = model_row
    # save back the dataframe
    df_results.to_csv(new_path, sep='\t', index=False)
print()

# now we can proceed dividing the model based on _ and creating the columns
columns_to_add = ['sim', 'nn', 'imp', 'bin', 'shrink', 'norm', 'asymalpha', 'tvalpha', 'tvbeta', 'rweights', 'samp-strat', 'val', 'n-h', 'n-t', 'sim-thres', 'w', 'nb', 'n-tr', 's-k', 'preposp']


for data in dataset:
    for exp in experiment:
        for cutoff in cutoffs:
            # find the path of the fixed file
            path = glob.glob(f'./results/{data}_{exp}/performance/rec_cutoff_fixed_{cutoff}_*.tsv')[0]
            final_path = path.replace('fixed', 'final')
            # read the file
            df_results = pd.read_csv(path, sep='\t')
            columns_old = df_results.columns
            # add the new columns between the model and the metric columns, we can define some default values for the new columns
            for col in columns_to_add:
                df_results[col] = None
            # reorder the columns
            new_order = [columns_old[0]] + columns_to_add + list(columns_old[1:])
            df_results = df_results[new_order]
            # iterate over the rows of the dataframe
            for i, row in df_results.iterrows():
                # take the first column
                model_row = row['model']
                # First, split by '_'
                parts = model_row.split('_')
                # save the model name in a variable
                model_name = parts[0]
                # we don't need anymore the name of the model
                parts = parts[1:]
                # Then, split each part by '='
                key_value_pairs = [part.split('=') for part in parts]
                # Convert into a dictionary (handling cases where there might be no '=')
                result_dict = {kv[0]: kv[1] if len(kv) > 1 else None for kv in key_value_pairs}
                # we can now add the values of kv[1] to the kv[0] column in the i-th row
                for key, value in result_dict.items():
                    df_results.loc[i, key] = value
                # the other values will remain to None
                # change the model name in the model column
                df_results.loc[i, 'model'] = model_name
            # in the sim_thres column, substitute all the $ values with .
            df_results['sim-thres'] = df_results['sim-thres'].str.replace('$', '.')
            # remove unnecessary columns
            unnecessary_columns = ['imp', 'bin', 'shrink', 'norm', 'asymalpha', 'tvalpha', 'tvbeta', 'rweights', 'val', 'sim-thres', 'w']
            df_results = df_results.drop(columns=unnecessary_columns)
            # before saving, we need to cast the parameters to the correct type
            dtype_mapping = {'model': str, 'sim': str, 'nn': int, 'samp-strat': str, 'n-h': int, 'n-t': int, 'nb': int, 'n-tr': int, 's-k': int, 'preposp': str}
            for k,v in dtype_mapping.items():
                df_results[k] = df_results[k].apply(lambda x: v(x) if pd.notna(x) else None)
            print()
            # save back the dataframe
            df_results.to_csv(final_path, sep='\t', index=False)




