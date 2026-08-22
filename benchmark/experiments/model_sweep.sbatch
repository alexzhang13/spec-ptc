#!/bin/bash
#SBATCH -p team4
#SBATCH -w node4
#SBATCH --gres=gpu:2
#SBATCH -J spec-ptc-sweep
#SBATCH --mem=300G
#SBATCH -o infra/model_sweep.%j.log
cd /shared/home/altzhang-de4f8c/spec-ptc
exec .venv/bin/python infra/model_sweep.py
