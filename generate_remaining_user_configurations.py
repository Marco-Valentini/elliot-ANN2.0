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

remaining_configurations = [("cosine", 2, 4, "approx_degree"),
                            ("cosine", 3, 8, "approx_degree"),
                            ("jaccard", 2, 1, "uniform"),
                            ("jaccard", 2, 8, "uniform"),
                            ("jaccard", 3, 1, "uniform"),
                            ("jaccard", 3, 8, "uniform"),
                            ("jaccard", 2, 8, "weighted_uniform"),
                            ("jaccard", 3, 1, "weighted_uniform"),
                            ("jaccard", 3, 8, "weighted_uniform"),
                            ("jaccard", 2, 1, "approx_degree"),
                            ("jaccard", 2, 2, "approx_degree"),
                            ("jaccard", 2, 8, "approx_degree"),
                            ("jaccard", 3, 1, "approx_degree"),
                            ("jaccard", 3, 8, "approx_degree"),
                            ("euclidean", 2, 1, "approx_degree"),
                            ("euclidean", 3, 1, "approx_degree"),
                            ("euclidean", 3, 2, "approx_degree"),
                            ("euclidean", 3, 4, "approx_degree"),
                            ("euclidean", 3, 2, "uniform"),
                            ("euclidean", 3, 2, "weighted_uniform")]

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
     neighbors: 500
     similarity: {similarity}
     sampling_strategy: {sampling_strategy}
     n_hash: {n_hash}
     n_tables: {n_tables}"""



for sim, n_hash, n_tables, sampling_strategy in remaining_configurations:
    with open(f"config_run_experiments_yelp/remaining_user_experiment_yelp_sim={sim}_samp_strat={sampling_strategy}_n_hash={n_hash}_n_tables={n_tables}.yml", "w") as f:
        f.write(template.format(similarity=sim, sampling_strategy=sampling_strategy, n_hash=n_hash, n_tables=n_tables))

# once generated the YAML files, we need to generate the sbatch files and run them on the cineca cluster
template_sbatch = """#!/bin/bash
#SBATCH --job-name=job_user_{similarity}_{sampling_strategy}_{n_hash}_{n_tables}
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=2
#SBATCH --output=user_remaining_out/log_user_{similarity}_{sampling_strategy}_{n_hash}_{n_tables}.out
#SBATCH --error=user_remaining_err/log_user_{similarity}_{sampling_strategy}_{n_hash}_{n_tables}.err
#SBATCH --account={account_no}
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m.valentini7@phd.poliba.it
#SBATCH --gres=gpu:0
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal

cd $WORK/elliot-ANN2.0/
module load anaconda3
source activate elliot_venv
python script_run_remaining_experiment_generic.py --type remaining_user --similarity {similarity} --sampling {sampling_strategy} --n_hash '{n_hash}' --n_tables '{n_tables}'"""

# create the directory for the sbatch files
os.makedirs("remaining_sbatch_files_user", exist_ok=True)
# submit a job for each configuration
for sim, n_hash, n_tables, sampling_strategy in remaining_configurations:
    # generate sbatch script content
    sbatch_content = template_sbatch.format(similarity=sim, sampling_strategy=sampling_strategy, n_hash=n_hash, n_tables=n_tables, account_no=account_no)
    # prepare the path for the sbatch file
    sbatch_file_path = f"remaining_sbatch_files_user/run_user_{sim}_{sampling_strategy}_{n_hash}_{n_tables}.sbatch"
    # write the sbatch file
    with open(sbatch_file_path, "w") as f:
        f.write(sbatch_content)
    print(f"Generated sbatch file {sbatch_file_path}")
    #
    # submit the job
    subprocess.run(["sbatch", sbatch_file_path])
    # add a delay to avoid submitting too many jobs at the same time
    time.sleep(5) # delay in seconds

