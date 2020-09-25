#!/bin/bash

GPU_ID=0
DOMAIN="furniture"
# DOMAIN="fashion"
ROOT="../data/simmc_${DOMAIN}/"


# Input files.
DEV_JSON_FILE="${ROOT}${DOMAIN}_dev_dials.json"
DEVTEST_JSON_FILE="${ROOT}${DOMAIN}_devtest_dials.json"


# Evaluate a trained model checkpoint.
CHECKPOINT_PATH="outputs/HAE_R300_lr3_b32_la3_0924_11:50/checkpoints/epoch_best_task2.tar"
python -u eval_simmc_agent.py \
    --eval_data_path=${DEVTEST_JSON_FILE/.json/_mm_inputs.npy} \
    --checkpoint="$CHECKPOINT_PATH" \
    --gpu_id=$GPU_ID \
    --batch_size=50 \
    --domain="$DOMAIN"
