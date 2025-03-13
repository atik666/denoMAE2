#!/bin/bash

# Activate the Python environment (replace `torch` with your actual environment name)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch

# Set paths for training and test data
TRAIN_PATH="/mnt/d/OneDrive - Rowan University/RA/Summer 24/MMAE_Wireless/DenoMAE/data/unlabeled_10k/train/"
TEST_PATH="/mnt/d/OneDrive - Rowan University/RA/Summer 24/MMAE_Wireless/DenoMAE/data/unlabeled_10k/test/"

# Set other configurations
BATCH_SIZE=64
IMAGE_SIZE="224 224"
PATCH_SIZE=16
EMBED_DIM=768
DECODER_EMBED_DIM=512 # it was 512 before. I changed it to 768
ENCODER_DEPTH=12
DECODER_DEPTH=8
ENCODER_NUM_HEADS=12
DECODER_NUM_HEADS=8
NUM_EPOCHS=100
LEARNING_RATE=3e-4
LOG_DIR="runs/denoMAE2"
MODEL_DIR="models"
NUM_MODALITY=1
MODEL_NAME="denoMAE2_MLP"
FINAL_MODEL_NAME="denoMAE2_MLp_final"
N_WORKERS=4
CLS_LOSS_RATIO=0.1
MASK_RATIO=0.75
TEST_BATCH_SIZE=10

# Run the Python script
python pretrain.py \
  --train_path "$TRAIN_PATH" \
  --test_path "$TEST_PATH" \
  --batch_size $BATCH_SIZE \
  --image_size $IMAGE_SIZE \
  --patch_size $PATCH_SIZE \
  --embed_dim $EMBED_DIM \
  --decoder_embed_dim $DECODER_EMBED_DIM \
  --encoder_depth $ENCODER_DEPTH \
  --decoder_depth $DECODER_DEPTH \
  --encoder_num_heads $ENCODER_NUM_HEADS \
  --decoder_num_heads $DECODER_NUM_HEADS \
  --num_epochs $NUM_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --log_dir $LOG_DIR \
  --model_dir $MODEL_DIR \
  --num_modality $NUM_MODALITY \
  --model_name $MODEL_NAME \
  --final_model_name $FINAL_MODEL_NAME \
  --n_workers $N_WORKERS \
  --cls_loss_ratio $CLS_LOSS_RATIO \
  --mask_ratio $MASK_RATIO \
  --test_batch_size $TEST_BATCH_SIZE
