#!/bin/bash

GPU_ID=$1
DOMAIN="furniture"
# DOMAIN="fashion"
ROOT="../data/simmc_${DOMAIN}/"


# Input files.
DEV_JSON_FILE="${ROOT}${DOMAIN}_dev_dials.json"
DEVTEST_JSON_FILE="${ROOT}${DOMAIN}_devtest_dials.json"

MODEL_LIST="
    HAE_R300 \
    HRE_G300_MAG \
    HRE_R300 \
    HRE_R300_MAG \
    HRE_R300_MMI \
    MN_G300_MAG \
    MN_R300_MAG \
    MN_R300_MMI"

for MODEL in $MODEL_LIST
do
    CHECKPOINT_PATH="outputs/${DOMAIN}/${MODEL}/checkpoints/"

    # Evaluate a trained model checkpoint.
    python -u inference_simmc_agent.py \
        --eval_data_path=${DEVTEST_JSON_FILE/.json/_mm_inputs.npy} \
        --task1_checkpoint="${CHECKPOINT_PATH}epoch_best_task1.tar" \
        --task2_g_checkpoint="${CHECKPOINT_PATH}epoch_best_task2_g.tar" \
        --task2_r_checkpoint="${CHECKPOINT_PATH}epoch_best_task2_r.tar" \
        --gpu_id=$GPU_ID \
        --batch_size=50 \
        --domain="$DOMAIN" \
        --pred_save_path="$CHECKPOINT_PATH"
done
