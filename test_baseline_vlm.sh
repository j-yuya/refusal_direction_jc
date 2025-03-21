#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=00:59:00
#SBATCH --job-name=vicuna_refusal_dir
#SBATCH --output=slurm/vicuna_refusal_dir_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --error=slurm/vicuna_refusal_dir_%j.err
#SBATCH --partition=gpu-vram-94gb
#SBATCH --gres=gpu:1

python3 baseline_test.py