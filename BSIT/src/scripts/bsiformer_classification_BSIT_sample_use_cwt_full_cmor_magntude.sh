#!/bin/bash -l
#SBATCH --chdir /scratch/izar/maetz
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 10
#SBATCH --mem 70G
#SBATCH --time 10:00:00
#SBATCH --gres gpu:1

# export CUDA_VISIBLE_DEVICES=0
python -u /src/train.py \
    --is_training 1 \
    --model_id "BSIformer_classification_BSIT_sample_use_cwt_cmor1-1_magnitude" \
    --model "BSIformerT" \
    --task "Classification" \
    --dataset "BSIsample" \
    --batch_size 512 \
    --embed_dim 32 \
    --hidden_dim 128 \
    --num_heads 2 \
    --num_layers 2 \
    --num_t_pints 744 \
    --num_patches 590 \
    --num_cut 1 \
    --n_epochs 150 \
    --early_stop 40 \
    --n_classes 4 \
    --plot_epoch 30 \
    --learning_rate 3e-4 \
    --weight_decay 2e-4 \
    --dropout 0.1 \
    --aug_variance 0.01 \
    --use_gpu True \
    --N_WORKERS 16 \
    --gpu "cuda:0" \
    --scaling \
    --mask_rate 0.0 \
    --etf \
    --use_cwt \
    # --use_fft \
    
    # --etf
    
