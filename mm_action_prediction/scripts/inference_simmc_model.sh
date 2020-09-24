#!/bin/bash

GPU_ID=0
DOMAIN="furniture"
# DOMAIN="fashion"
ROOT="../data/simmc_${DOMAIN}/"


# Input files.
DEV_JSON_FILE="${ROOT}${DOMAIN}_dev_dials.json"
DEVTEST_JSON_FILE="${ROOT}${DOMAIN}_devtest_dials.json"

SAVE_PATH="outputs/test_a_0831_00:14/checkpoints/"
CHECKPOINT_FILE="transf/epoch_best.tar"
CHECKPOINT_PATH="${SAVE_PATH}${CHECKPOINT_FILE}"

# Evaluate a trained model checkpoint.
python -u eval_simmc_agent.py \
    --eval_data_path=${DEVTEST_JSON_FILE/.json/_mm_inputs.npy} \
    --checkpoint="$CHECKPOINT_PATH" \
    --gpu_id=$GPU_ID \
    --batch_size=50 \
    --domain="$DOMAIN" \
    --test \
    --pred_save_path="${SAVE_PATH}"
