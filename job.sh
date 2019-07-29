#!/bin/bash#SBATCH --qos=unkillable # Ask for unkillable job#SBATCH --cpus-per-task=2 # Ask for 2 CPUs#SBATCH --gres=gpu:1 # Ask for 1 GPU#SBATCH --mem=10G # Ask for 10 GB of RAM#SBATCH --time=3:00:00 # The job will run for 3 hours#milacluster 4/4#SBATCH --time=3:00:00 # The job will run for 3 hours#SBATCH -o /network/tmp1/<user>/slurm-%j.out # Write the log on tmp1# 1. Load your environment# 2. Copy your dataset on the compute node# 3. Launch your job, tell it to save the model in $SLURM_TMPDIR# and look for the dataset into $SLURM_TMPDIRexport PYTHONPATH=/network/home/gomrokma/nips_paperpython3 main.py# 4. Copy whatever you want to save on $SCRATCH