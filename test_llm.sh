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

#python3 -m pipeline.run_pipeline --model_path /ceph/jcaspary/hf_cache/hub/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d
python3 -m pipeline.run_pipeline --model_path /ceph/jcaspary/hf_cache/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590