# this script takes the recommendation files and divides it into different folders, each containing 30 recommendation files

# import the requried libraries
import os
import shutil
from math import ceil
import warnings
import argparse
import subprocess
import time

# Suppress all warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Generate the sbatch files for the item-based and user-based experiments with FairANN.")
parser.add_argument('--account', type=str)
args = parser.parse_args()

account_no = args.account


# define the paths

item_recs_path = "results/lastfm_1k_item_ed/recs/"
user_recs_path = "results/lastfm_1k_user_ed/recs/"

# find how many files are there in the item recommendation folder
item_files = os.listdir(item_recs_path)
print(f"Total item files: {len(item_files)}")

# find how many files are there in the user recommendation folder
user_files = os.listdir(user_recs_path)
print(f"Total user files: {len(user_files)}")

# find the number of folder to be created
n_folders = ceil(len(item_files) / 20)

# create the folders into the item folder
for i in range(n_folders):
    os.makedirs(f"results/lastfm_1k_item_ed/recs_{i+1}", exist_ok=True)

# create the folders into the user folder
for i in range(n_folders):
    os.makedirs(f"results/lastfm_1k_user_ed/recs_{i+1}", exist_ok=True)

# move the files into the folders
for i, file in enumerate(item_files):
    folder_no = i // 20
    shutil.move(f"{item_recs_path}{file}", f"results/lastfm_1k_item_ed/recs_{folder_no+1}/{file}")

for i, file in enumerate(user_files):
    folder_no = i // 20
    shutil.move(f"{user_recs_path}{file}", f"results/lastfm_1k_user_ed/recs_{folder_no+1}/{file}")

print("Done!")

template_item_yaml = """experiment:
  dataset: lastfm_1k
  data_config:
    strategy: fixed
    train_path: ../data/lastfm_1k/filtered_data/0/train.tsv
    test_path: ../data/lastfm_1k/filtered_data/0/test.tsv
  top_k: 50
  path_output_rec_result: ./results/lastfm_1k_item_ed/recs_{i}/
  path_output_rec_weight: ./results/lastfm_1k_item_ed/weights_{i}/
  path_output_rec_performance: ./results/lastfm_1k_item_ed/performance_{i}/
  path_log_folder: ../log/lastfm_1k_item_ed_{i}/
  evaluation:
    cutoffs: [ 1,5,10,20 ]
    simple_metrics: [ nDCGRendle2020, Recall, HR, Precision, MAP, MRR, ItemCoverage, ACLT, Gini, SEntropy, EFD, EPC, PopREO, PopRSP ]
    complex_metrics:
      - metric: BiasDisparityBD
        user_clustering_name: UserTolerance
        user_clustering_file: ../data/lastfm_1k/users_tolerance_2.tsv
        item_clustering_name: ItemPopularity
        item_clustering_file: ../data/lastfm_1k/items_popularity_2.tsv
      - metric: BiasDisparityBR
        user_clustering_name: UserTolerance
        user_clustering_file: ../data/lastfm_1k/users_tolerance_2.tsv
        item_clustering_name: ItemPopularity
        item_clustering_file: ../data/lastfm_1k/items_popularity_2.tsv
      - metric: BiasDisparityBS
        user_clustering_name: UserTolerance
        user_clustering_file: ../data/lastfm_1k/users_tolerance_2.tsv
        item_clustering_name: ItemPopularity
        item_clustering_file: ../data/lastfm_1k/items_popularity_2.tsv
      - metric: REO
        clustering_name: ItemPopularity
        clustering_file: ../data/lastfm_1k/items_popularity_2.tsv
      - metric: RSP
        clustering_name: ItemPopularity
        clustering_file: ../data/lastfm_1k/items_popularity_2.tsv
  gpu: 0
  models:
    RecommendationFolder:
        folder: ./results/lastfm_1k_item_ed/recs_{i}/"""

template_user_yaml = """experiment:
  dataset: lastfm_1k
  data_config:
    strategy: fixed
    train_path: ../data/lastfm_1k/filtered_data/0/train.tsv
    test_path: ../data/lastfm_1k/filtered_data/0/test.tsv
  top_k: 50
  path_output_rec_result: ./results/lastfm_1k_user_ed/recs_{i}/
  path_output_rec_weight: ./results/lastfm_1k_user_ed/weights_{i}/
  path_output_rec_performance: ./results/lastfm_1k_user_ed/performance_{i}/
  path_log_folder: ../log/lastfm_1k_user_ed_{i}/
  evaluation:
    cutoffs: [ 1,5,10,20 ]
    simple_metrics: [ nDCGRendle2020, Recall, HR, Precision, MAP, MRR, ItemCoverage, ACLT, Gini, SEntropy, EFD, EPC, PopREO, PopRSP ]
    complex_metrics:
      - metric: BiasDisparityBD
        user_clustering_name: UserTolerance
        user_clustering_file: ../data/lastfm_1k/users_tolerance_2.tsv
        item_clustering_name: ItemPopularity
        item_clustering_file: ../data/lastfm_1k/items_popularity_2.tsv
      - metric: BiasDisparityBR
        user_clustering_name: UserTolerance
        user_clustering_file: ../data/lastfm_1k/users_tolerance_2.tsv
        item_clustering_name: ItemPopularity
        item_clustering_file: ../data/lastfm_1k/items_popularity_2.tsv
      - metric: BiasDisparityBS
        user_clustering_name: UserTolerance
        user_clustering_file: ../data/lastfm_1k/users_tolerance_2.tsv
        item_clustering_name: ItemPopularity
        item_clustering_file: ../data/lastfm_1k/items_popularity_2.tsv
      - metric: REO
        clustering_name: ItemPopularity
        clustering_file: ../data/lastfm_1k/items_popularity_2.tsv
      - metric: RSP
        clustering_name: ItemPopularity
        clustering_file: ../data/lastfm_1k/items_popularity_2.tsv
  gpu: 0
  models:
    RecommendationFolder:
        folder: ./results/lastfm_1k_user_ed/recs_{i}/"""

# create and save the configuration files
for i in range(1, n_folders+1):
    with open(f"config_evaluate_exact_degree/evaluate_item_ed_lastfm_1k_{i}.yml", "w") as f:
        f.write(template_item_yaml.format(i=i))
    with open(f"config_evaluate_exact_degree/evaluate_user_ed_lastfm_1k_{i}.yml", "w") as f:
        f.write(template_user_yaml.format(i=i))

print("Done!")

# build a template for the sbatch file
template_sbatch_item = """#!/bin/bash
#SBATCH --job-name=job_evaluate_item_ed_{i}
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=1
#SBATCH --output=evaluate_out/evaluate_item_ed_{i}.out
#SBATCH --error=evaluate_err/evaluate_item_ed_{i}.err
#SBATCH --account={account_no}
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m.valentini7@phd.poliba.it
#SBATCH --gres=gpu:0
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal

cd $WORK/elliot-ANN2.0/
module load anaconda3
source activate elliot_venv
python script_run_generic.py --config_path 'config_evaluate_exact_degree/evaluate_item_ed_lastfm_1k_{i}.yml'
"""

# build a template for the sbatch file
template_sbatch_user = """#!/bin/bash
#SBATCH --job-name=job_evaluate_user_ed_{i}
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=1
#SBATCH --output=evaluate_out/evaluate_user_ed_{i}.out
#SBATCH --error=evaluate_err/evaluate_user_ed_{i}.err
#SBATCH --account={account_no}
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m.valentini7@phd.poliba.it
#SBATCH --gres=gpu:0
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal

cd $WORK/elliot-ANN2.0/
module load anaconda3
source activate elliot_venv
python script_run_generic.py --config_path 'config_evaluate_exact_degree/evaluate_user_ed_lastfm_1k_{i}.yml'
"""

# create the directory for the sbatch files
os.makedirs("sbatch_files_evaluate_item_ed", exist_ok=True)
os.makedirs("sbatch_files_evaluate_user_ed", exist_ok=True)

for i in range(n_folders+1):
    # generate sbatch script content for the item
    sbatch_content_item = template_sbatch_item.format(i=i+1, account_no=account_no)
    # prepare the path for the sbatch file for the item
    sbatch_file_path_item = f"sbatch_files_evaluate_item_ed/run_evaluate_item_ed_{i+1}.sbatch"
    # write the sbatch file for the item
    with open(sbatch_file_path_item, "w") as f:
        f.write(sbatch_content_item)
    print(f"Generated sbatch file {sbatch_file_path_item}")
    # generate sbatch script content for the user
    sbatch_content_user = template_sbatch_user.format(i=i+1, account_no=account_no)
    # prepare the path for the sbatch file for the user
    sbatch_file_path_user = f"sbatch_files_evaluate_user_ed/run_evaluate_user_ed_{i+1}.sbatch"
    # write the sbatch file for the user
    with open(sbatch_file_path_user, "w") as f:
        f.write(sbatch_content_user)
    print(f"Generated sbatch file {sbatch_file_path_user}")

    # submit the job
    subprocess.run(["sbatch", sbatch_file_path_item])
    # add a delay to avoid submitting too many jobs at the same time
    time.sleep(5) # delay in seconds
    # submit the job
    subprocess.run(["sbatch", sbatch_file_path_user])
    # add a delay to avoid submitting too many jobs at the same time
    time.sleep(5) # delay in seconds
