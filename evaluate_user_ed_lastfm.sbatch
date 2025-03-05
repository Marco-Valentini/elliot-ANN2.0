#!/bin/bash
#SBATCH --job-name=job_evaluate_user_ed_lastfm
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=1
#SBATCH --output=evaluate_user_ed_lastfm.out
#SBATCH --error=evaluate_user_ed_lastfm.err
#SBATCH --account=None
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m.valentini7@phd.poliba.it
#SBATCH --gres=gpu:0
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprof

cd $WORK/elliot-ANN2.0/
module load anaconda3
source activate elliot_venv
python script_run_generic.py --config_file config_evaluate_exact_degree/evaluate_user_lastfm_1k.yml