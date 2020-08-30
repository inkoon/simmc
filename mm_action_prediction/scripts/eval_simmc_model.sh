#!/bin/bash

GPU_ID=0
DOMAIN="furniture"
# DOMAIN="fashion"
ROOT="../data/simmc_${DOMAIN}/"


# Input files.
DEV_JSON_FILE="${ROOT}${DOMAIN}_dev_dials.json"
DEVTEST_JSON_FILE="${ROOT}${DOMAIN}_devtest_dials.json"


# Evaluate a trained model checkpoint.
CHECKPOINT_PATH="checkpoints/baselines/hae/epoch_30.tar"
python -u eval_simmc_agent.py \
    --eval_data_path=${DEV_JSON_FILE/.json/_mm_inputs.npy} \
    --checkpoint="$CHECKPOINT_PATH" \
    --gpu_id=$GPU_ID \
    --batch_size=50 \
    --domain="$DOMAIN"
