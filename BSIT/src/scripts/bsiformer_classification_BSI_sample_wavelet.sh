#!/bin/bash -l
#SBATCH --chdir /scratch/izar/maetz
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 10
#SBATCH --mem 70G
#SBATCH --time 00:05:00
#SBATCH --gres gpu:1

# export CUDA_VISIBLE_DEVICES=0
python -u /src/train.py \
    --is_training 1 \
    --model_id "BSIformer_classification_BSI_samplewavelet2_relu_cut1_emb32_channelemb_linearattn2_4class_linear_batchstand_etf5" \
    --model "BSIformer" \
    --task "Classification" \
    --dataset "BSIsamplewavelet" \
    --seed 123 \
    --batch_size 250 \
    --embed_dim 32 \
    --hidden_dim 128 \
    --num_heads 2 \
    --num_layers 2 \
    --num_t_pints 240 \
    --num_patches 32 \
    --num_cut 1 \
    --n_epochs 250 \
    --early_stop 50 \
    --n_classes 4 \
    --plot_epoch 30 \
    --learning_rate 3e-4 \
    --weight_decay 2e-4 \
    --dropout 0.1 \
    --aug_variance 0.05 \
    --use_gpu True \
    --N_WORKERS 16 \
    --gpu "cuda:0" \
    --mask_rate 0 \
    --scaling \
    --etf \
    # --use_fft \
    
    # --etf
    
