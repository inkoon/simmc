#!/bin/bash

# Check the parameters
if [ $# -lt 1 ];then
    echo "There's no parameters -> ./run.sh [gpu no]"
    exit 1
fi

GPU_ID=$1
DOMAIN="furniture"
# DOMAIN="fashion"
ROOT="../data/belief_simmc_${DOMAIN}/"


# Input files.
DEV_JSON_FILE="${ROOT}${DOMAIN}_dev_dials.json"
DEVTEST_JSON_FILE="${ROOT}${DOMAIN}_devtest_dials.json"
TESTSTD_JSON_FILE="${ROOT}${DOMAIN}_teststd_dials.json"

MODEL="HRE_R300_S_TD"

CHECKPOINT_PATH="outputs/${DOMAIN}/${MODEL}/checkpoints/"

# Evaluate a trained model checkpoint.
python -u inference_simmc_agent.py \
    --eval_data_path=${TESTSTD_JSON_FILE/.json/_mm_inputs.npy} \
    --task1_checkpoint="${CHECKPOINT_PATH}epoch_best_task1.tar" \
    --task2_g_checkpoint="${CHECKPOINT_PATH}epoch_best_task2_g.tar" \
    --task2_r_checkpoint="${CHECKPOINT_PATH}epoch_best_task2_r.tar" \
    --gpu_id=$GPU_ID \
    --embedding_type="random"\
    --gate_type="none" \
    --use_belief_state \
    --use_task3_belief_state \
    --batch_size=50 \
    --domain="$DOMAIN" \
    --pred_save_path="$CHECKPOINT_PATH" \
    --inference
