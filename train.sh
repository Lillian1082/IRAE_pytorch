#!/usr/bin/env bash

python3 -u IRAE_train_denoise.py \
    --train \
    --denoise \
    --archi=IRAE \
    --device_ids=1 \
    --batch_size=100 \
    --world_size=1 \
    --log_interval=1 \
    --lr=1e-3 \
    --width=64 \
    --depth=16 \
    --n_levels=2 \
    --n_epochs=150 \
    --milestone=50 \
    --dataset=celeba \
    --data_dir=data \
    --noise_mode='S' \
    --noise_level=15 \
    --n_epochs_warmup=-1

python3 -u GAE_train_inpainting.py \
    --train \
    --inpainting \
    --inpainting_mode=center \
    --archi=GAE \
    --device_ids=1 \
    --batch_size=18 \
    --world_size=1 \
    --log_interval=1 \
    --lr=1e-4 \
    --width=64 \
    --depth=8 \
    --n_levels=3 \
    --n_epochs=200 \
    --milestone=50 \
    --dataset=celeba_HQ \
    --data_dir=data \
    --n_epochs_warmup=-1