#!/bin/bash -l
#SBATCH --chdir /scratch/izar/maetz
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 70G
#SBATCH --time 00:30:00
#SBATCH --gres gpu:1

python /original_model/python_scripts/preprocessed_prediction.py \
    --is_training 1\
    --model_name preprocessed_prediction\

