# Starting from a string taken as template, this script generates a set YAML configuration files for the user-based
# experiments with Elliot

import warnings
import os
import argparse
import subprocess
import time

# Suppress all warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Generate the sbatch files for the remaining user-based experiments with ANNOY.")
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
    UserANNOY:
      meta:
        verbose: True
        save_recs: True
        validation_metric: nDCGRendle2020@20
        hyper_opt_alg: grid
      neighbors: {neighbors}
      similarity: angular
      n_trees: {n_trees}
      search_k: {search_k}
"""

neighbors = [250]
n_trees = [1, 3, 5]
search_k = [-1, 5, 10]

for k in neighbors:
    for t in n_trees:
        for s in search_k:
            if k == 250 and t == 5 and s == 10:
                continue
            else:
                with open(f"config_run_experiments_yelp/remaining_user_ANNOY_experiment_yelp_neighbors={k}_n_trees={t}_search_k={s}.yml", "w") as f:
                    f.write(template.format(neighbors=k, n_trees=t, search_k=s))

# once generated the YAML files, we need to generate the sbatch files and run them on the cineca cluster
template_sbatch = """#!/bin/bash
#SBATCH --job-name=job_annoy_user_n={neighbors}_t={n_trees}_k={search_k}
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=1
#SBATCH --output=user_remaining_annoy_out/log_annoy_user_n={neighbors}_t={n_trees}_k={search_k}.out
#SBATCH --error=user_remaining_annoy_err/log_annoy_user_n={neighbors}_t={n_trees}_k={search_k}.err
#SBATCH --account={account_no}
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m.valentini7@phd.poliba.it
#SBATCH --gres=gpu:0
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal

cd $WORK/elliot-ANN2.0/
module load anaconda3
source activate elliot_venv
python script_run_experiment_annoy_generic.py --type remaining_user --neighbors {neighbors} --n_trees {n_trees} --search_k {search_k}
"""
# create the directory for the sbatch files
os.makedirs("remaining_sbatch_files_user_ANNOY", exist_ok=True)
# submit a job for each configuration
for k in neighbors:
    for t in n_trees:
        for s in search_k:
            if k == 250 and t == 5 and s == 10:
                continue
            else:
                # generate sbatch script content
                sbatch_content = template_sbatch.format(neighbors=k, n_trees=t, search_k=s, account_no=account_no)
                # prepare the path for the sbatch file
                sbatch_file_path = f"remaining_sbatch_files_user_ANNOY/run_annoy_user_n={k}_t={t}_k={s}.sbatch"
                # write the sbatch file
                with open(sbatch_file_path, "w") as f:
                    f.write(sbatch_content)
                print(f"Generated sbatch file {sbatch_file_path}")

                # submit the job
                subprocess.run(["sbatch", sbatch_file_path])
                # add a delay to avoid submitting too many jobs at the same time
                time.sleep(5)  # delay in seconds
