#!/bin/bash -l
#SBATCH --chdir /scratch/izar/maetz
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 70G
#SBATCH --time 30:00:00
#SBATCH --gres gpu:1

python /original_model/python_scripts/upsampled_dwt_prediction.py \
    --is_training 1\
    --model_name upsampled_dwt_prediction\
    --preprocessing 1\
    --batch_size 16\
    --max_epochs 50\

