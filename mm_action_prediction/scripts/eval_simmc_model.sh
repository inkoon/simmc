#!/bin/bash

GPU_ID=0
DOMAIN="furniture"
# DOMAIN="fashion"
ROOT="../data/simmc_${DOMAIN}/"


# Input files.
DEV_JSON_FILE="${ROOT}${DOMAIN}_dev_dials.json"
DEVTEST_JSON_FILE="${ROOT}${DOMAIN}_devtest_dials.json"


# Evaluate a trained model checkpoint.
CHECKPOINT_PATH="outputs/test_a_0831_00:14/checkpoints/transf/epoch_best.tar"
python -u eval_simmc_agent.py \
    --eval_data_path=${DEVTEST_JSON_FILE/.json/_mm_inputs.npy} \
    --checkpoint="$CHECKPOINT_PATH" \
    --gpu_id=$GPU_ID \
    --batch_size=50 \
    --domain="$DOMAIN"
