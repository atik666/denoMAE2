#!/bin/bash

# Activate the Python environment (replace `torch` with your actual environment name)
source activate torch || source ~/.bashrc && conda activate torch

# Define the dB value
DB_VALUE="10"  # Change this value as needed

# Define the paths to training and testing data using the DB_VALUE
# TRAIN_DATA_PATH="/mnt/d/OneDrive - Rowan University/RA/Summer 24/MMAE_Wireless/DenoMAE/data/labeled/${DB_VALUE}_dB/train/"
# TEST_DATA_PATH="/mnt/d/OneDrive - Rowan University/RA/Summer 24/MMAE_Wireless/DenoMAE/data/labeled/${DB_VALUE}_dB/test/"

# TRAIN_DATA_PATH="/mnt/d/OneDrive - Rowan University/RA/Summer 24/MMAE_Wireless/DenoMAE/data/labeled_mix/train/"
# TRAIN_DATA_PATH="/mnt/d/OneDrive - Rowan University/RA/Summer 24/MMAE_Wireless/DenoMAE/data/labeled_mix/test/"

TRAIN_DATA_PATH="/mnt/d/archive/images_large/snr_20db/train/"
TEST_DATA_PATH="/mnt/d/archive/images_large/snr_20db/test/"
# Define the pretrained model path and output model path
PRETRAINED_MODEL_PATH="models/denoMAE2.pth"
OUTPUT_MODEL_PATH="models/finetunedClassifier.pth"  # Added dB value to output filename

# Run the Python script with the specified arguments
python finetune.py \
    --train_data_path "$TRAIN_DATA_PATH" \
    --test_data_path "$TEST_DATA_PATH" \
    --image_size 224 224 \
    --patch_size 16 \
    --embed_dim 768 \
    --decoder_embed_dim 512 \
    --encoder_depth 12 \
    --decoder_depth 8 \
    --encoder_num_heads 12 \
    --decoder_num_heads 8 \
    --batch_size 10 \
    --num_epochs 150 \
    --learning_rate 1e-4 \
    --num_classes 24 \
    --pretrained_model_path "$PRETRAINED_MODEL_PATH" \
    --output_model_path "$OUTPUT_MODEL_PATH" \
    --gpu "0,1" \