# Starting from a string taken as template, this script generates a set YAML configuration files for the user-based
# experiments with Elliot

import warnings
import os
import argparse
import subprocess
import time

# Suppress all warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Run a script to run experiments on a generic dataset.")
parser.add_argument('--account', type=str)
args = parser.parse_args()

account_no = args.account

template = """experiment:
  dataset: yelp
  data_config:
    strategy: fixed
    train_path: ../data/yelp/filtered_data/0/train.tsv
    test_path: ../data/yelp/filtered_data/0/test.tsv
    side_information:
      - dataloader: ItemPopularityUserTolerance
        user_file: ../data/yelp/users_tolerance_2.tsv
        item_file: ../data/yelp/items_popularity_2.tsv
        user_group_file: ../data/yelp/group_tolerance_2_to_users.json
        item_group_file: ../data/yelp/group_popularity_2_to_items.json
  top_k: 50
  path_output_rec_result: ./results/yelp_user/recs/
  path_output_rec_weight: ./results/yelp_user/weights/
  path_output_rec_performance: ./results/yelp_user/performance/
  path_log_folder: ../log/yelp_user/
  evaluation:
    cutoffs: 20
    simple_metrics: [ nDCGRendle2020 ]
  external_models_path: ../external/models/__init__.py
  models:
   UserFairANN:
     meta:
       verbose: True
       save_recs: True
       validation_metric: nDCGRendle2020@20
       hyper_opt_alg: grid
     neighbors: {neighbors}
     similarity: [ cosine, jaccard, euclidean ]
     sampling_strategy: {sampling_strategy}
     n_hash: {n_hash}
     n_tables: [ 1,2,4,8 ]"""

n_hash = [ [2, 3], [4, 5] ]
sampling_strategy = [ "no_sampling", "uniform", "weighted_uniform", "opt", "approx_degree", "rank" ]
neighbors = [ 50, 100, 250, 500 ]

for n in n_hash:
    for s in sampling_strategy:
        for k in neighbors:
            with open(f"config_run_experiments_yelp/user_experiment_yelp_n_hash={n}_sampling={s}_neighbors={k}.yml", "w") as f:
                f.write(template.format(n_hash=n, sampling_strategy=s, neighbors=k))

# once generated the YAML files, we need to generate the sbatch files and run them on the cineca cluster
template_sbatch = """#!/bin/bash
#SBATCH --job-name=job_user_{n_hash}_{sampling_strategy}_{neighbors}
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=2
#SBATCH --output=log_user_{n_hash}_{sampling_strategy}_{neighbors}.out
#SBATCH --error=log_user_{n_hash}_{sampling_strategy}_{neighbors}.err
#SBATCH --account={account_no}
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m.valentini7@phd.poliba.it
#SBATCH --gres=gpu:0
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal

module load anaconda3
source activate elliot_venv
python script_run_experiment_generic.py --type user --n_hash '{n_hash}' --sampling {sampling_strategy} --neighbors {neighbors}
"""
i = 1
# create the directory for the sbatch files
os.makedirs("sbatch_files_user", exist_ok=True)
# submit a job for each configuration
for n in n_hash:
    for s in sampling_strategy:
        for k in neighbors:
            # generate sbatch script content
            sbatch_content = template_sbatch.format(n_hash=n, sampling_strategy=s, neighbors=k, account_no=account_no)
            # prepare the path for the sbatch file
            sbatch_file_path = f"sbatch_files_user/run_user_{i}.sbatch"
            i += 1
            # write the sbatch file
            with open(sbatch_file_path, "w") as f:
                f.write(sbatch_content)
            print(f"Generated sbatch file {sbatch_file_path}")

            # submit the job
            subprocess.run(["sbatch", sbatch_file_path])
            # add a delay to avoid submitting too many jobs at the same time
            time.sleep(60) # delay in seconds

