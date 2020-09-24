#!/bin/bash

GPU_ID=0
DOMAIN="furniture"
# DOMAIN="fashion"
ROOT="../data/simmc_${DOMAIN}/"


# Input files.
DEV_JSON_FILE="${ROOT}${DOMAIN}_dev_dials.json"
DEVTEST_JSON_FILE="${ROOT}${DOMAIN}_devtest_dials.json"

CHECKPOINT_PATH="outputs/test_a_0831_00:14/checkpoints/"
CHECKPOINT_FILE="transf/epoch_best.tar"

# Evaluate a trained model checkpoint.
python -u eval_simmc_agent.py \
    --eval_data_path=${DEVTEST_JSON_FILE/.json/_mm_inputs.npy} \
    --task1_checkpoint="${CHECKPOINT_PATH}epoch_best_task1.tar" \
    --task2_checkpoint="${CHECKPOINT_PATH}epoch_best_task2.tar" \
    --gpu_id=$GPU_ID \
    --batch_size=50 \
    --domain="$DOMAIN" \
    --pred_save_path="$CHECKPOINT_PATH"
