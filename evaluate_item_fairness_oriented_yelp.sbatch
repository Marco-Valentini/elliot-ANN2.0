#!/bin/bash
#SBATCH --job-name=job_evaluate_item_fairness_oriented_yelp
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=1
#SBATCH --output=evaluate_item_fairness_oriented.out
#SBATCH --error=evaluate_item_fairness_oriented.err
#SBATCH --account=None
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m.valentini7@phd.poliba.it
#SBATCH --gres=gpu:0
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal

cd $WORK/elliot-ANN2.0/
module load anaconda3
source activate elliot_venv
python script_evaluate_generic_fairness.py --type item --dataset yelp