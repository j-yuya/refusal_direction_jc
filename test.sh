#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=02:59:00
#SBATCH --job-name=vicuna_refusal_dir
#SBATCH --output=slurm/vicuna_refusal_dir_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --error=slurm/vicuna_refusal_dir_%j.err
#SBATCH --partition=gpu-vram-32gb
#SBATCH --gres=gpu:1

python3 -m pipeline.run_pipeline --model_path /ceph/jcaspary/hf_cache/hub/models--TRI-ML--prismatic-vlms/snapshots/a3ba8a19c453a82eaf5a3fb1e699dd9e441f0a12/reproduction-llava-v15+7b