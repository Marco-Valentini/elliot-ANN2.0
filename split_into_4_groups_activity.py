import pandas as pd
import numpy as np
import argparse
import json

parser = argparse.ArgumentParser(description="Run split user/item groups depending on quartiles.")
parser.add_argument('--dataset', type=str, default='movielens_100k', choices=['movielens_100k', 'movielens_1m', 'amazon_books', 'lastfm-1K'])
args = parser.parse_args()
# take the dataset from the input arguments
dataset = args.dataset
# read the dataset from the corresponding csv file
df = pd.read_csv(f'./data/{dataset}/filtered_data/0/train.tsv', sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
df = df.drop('timestamp', axis=1)

print("data successfully loaded")
# count the transactions in which any user/item occurs
user_count = df['user_id'].value_counts()
item_count = df['item_id'].value_counts()

print("Sorting the values...")
user_frequencies = pd.DataFrame(data=np.concatenate([user_count.index.values.reshape(-1,1), user_count.values.reshape(-1,1)], axis=1)).sort_values(by=1, ascending=True)
user_frequencies.columns = ['UserID', 'Frequency']
item_frequencies = pd.DataFrame(data=np.concatenate([item_count.index.values.reshape(-1,1), item_count.values.reshape(-1,1)], axis=1)).sort_values(by=1, ascending=True)
item_frequencies.columns = ['ItemID', 'Frequency']


print("Computing the quantiles for the distributions...")
user_25 = np.quantile(user_frequencies['Frequency'], 0.25)
user_50 = np.quantile(user_frequencies['Frequency'], 0.50)
user_75 = np.quantile(user_frequencies['Frequency'], 0.75)
item_25 = np.quantile(item_frequencies['Frequency'], 0.25)
item_50 = np.quantile(item_frequencies['Frequency'], 0.50)
item_75 = np.quantile(item_frequencies['Frequency'], 0.75)

user_bins = [0, user_25, user_50, user_75, np.inf]
user_bins = list(set(user_bins))
user_bins.sort()

user_labels = [i for i in range(len(user_bins)-1)]

item_bins = [0, item_25, item_50, item_75, np.inf]
item_bins = list(set(item_bins))
item_bins.sort()
item_labels = [i for i in range(len(item_bins)-1)]

print("Assigning the labels to each bin")
user_frequencies['group'] = pd.cut(user_frequencies['Frequency'], bins=user_bins, labels=user_labels, right=True)
item_frequencies['group'] = pd.cut(item_frequencies['Frequency'], bins=item_bins, labels=item_labels, right=True)

# create a column for when the users/items are not split
user_frequencies['no_group'] = np.full(shape=user_frequencies.shape[0], fill_value=0)
item_frequencies['no_group'] = np.full(shape=item_frequencies.shape[0], fill_value=0)

# create a reverse mapping groupID: [list of users/items belonging to that group]
group_to_users = user_frequencies.groupby('group')['UserID'].apply(list).to_dict()
group_to_items = item_frequencies.groupby('group')['ItemID'].apply(list).to_dict()

print("Saving the results...")
# save the grouping of ID-group
user_frequencies[['UserID', 'group']].to_csv(f'data/{dataset}/user_groups_4.tsv', sep='\t', header=None, index=None)
item_frequencies[['ItemID', 'group']].to_csv(f'data/{dataset}/item_groups_4.tsv', sep='\t', header=None, index=None)

# save the dataset without grouping
user_frequencies[['UserID', 'no_group']].to_csv(f'data/{dataset}/user_no_groups.tsv', sep='\t', header=None, index=None)
item_frequencies[['ItemID', 'no_group']].to_csv(f'data/{dataset}/item_no_groups.tsv', sep='\t', header=None, index=None)

# save the reverse mapping groupID -> list of User/item IDs
with open(f'data/{dataset}/group4_to_users.json', 'w') as file:
    json.dump(group_to_users, file, indent=4)

with open(f'data/{dataset}/group4_to_items.json', 'w') as file:
    json.dump(group_to_items, file, indent=4)


# group labelled as 3 contains the most popular/most active users (25%), the group labelled as 0-1-2 contains users
# and items from the long-tail (75%)