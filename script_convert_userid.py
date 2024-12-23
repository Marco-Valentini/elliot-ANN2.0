import pandas as pd
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Convert string user IDs to numeric IDs in a TSV file")
parser.add_argument('--input_file', type=str, help="Input TSV file")

args = parser.parse_args()
input_file = "data/" + args.input_file + "/dataset.tsv"

# Read the TSV file into a pandas DataFrame
df = pd.read_csv(input_file, sep='\t', names=['UserID', 'ItemID', 'Rating', 'Timestamp'])

# Map unique UserID strings to numeric IDs
unique_ids = {user: idx for idx, user in enumerate(df['UserID'].unique())}

# Add a new column with numeric IDs
df['NumericUserID'] = df['UserID'].map(unique_ids)

# Remove the old UserID column and rename the numeric ID column
df = df.drop(columns=['UserID'])
df = df.rename(columns={'NumericUserID': 'UserID'})

# Reorder columns to original order
df = df[['UserID', 'ItemID', 'Rating', 'Timestamp']]

# Save the updated DataFrame back to same TSV file
df.to_csv(input_file, sep='\t', index=None, header=None)