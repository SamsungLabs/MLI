#!/bin/bash

#NVIDIA_VISIBLE_DEVICES=1

#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=1234 bin/train.py --config configs/tblock4_train.yaml --output-path tblock_4_1gpu
CUDA_VISIBLE_DEVICES=0 torchrun \
    --standalone \
    --nproc_per_node=1 \
    bin/train.py \
    --config configs/tblock4_train_local.yaml \
    --output-path tblock_4_2gpu_new3