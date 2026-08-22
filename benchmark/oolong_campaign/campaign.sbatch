#!/bin/bash
#SBATCH -p team4
#SBATCH -w node4
#SBATCH --mem=400G
#SBATCH --gres=gpu:2
#SBATCH -J oolong-campaign
#SBATCH -o benchmark/oolong_campaign/campaign.%j.log
cd /shared/home/altzhang-de4f8c/spec-ptc
exec .venv/bin/python benchmark/oolong_campaign/campaign.py ${1:-main}
