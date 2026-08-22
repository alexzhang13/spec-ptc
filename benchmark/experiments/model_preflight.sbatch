#!/bin/bash
#SBATCH -p team4
#SBATCH -w node4
#SBATCH --mem=300G
#SBATCH --gres=gpu:2
#SBATCH -J spec-preflight
#SBATCH -o benchmark/experiments/preflight.%j.log
cd /shared/home/altzhang-de4f8c/spec-ptc
exec .venv/bin/python benchmark/experiments/model_preflight.py
