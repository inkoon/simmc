#!/bin/bash

# Check the parameters
if [ $# -lt 2 ];then
    echo "There's no parameters -> ./run.sh [model descriptions] [gpu no]"
    exit 1
fi

DOMAIN="furniture"
#DOMAIN="fashion"
ROOT="../data/simmc_${DOMAIN}/"
DETAILS=$1
GPU_ID=$2


# Input files.
TRAIN_JSON_FILE="${ROOT}${DOMAIN}_train_dials.json"
DEV_JSON_FILE="${ROOT}${DOMAIN}_dev_dials.json"
DEVTEST_JSON_FILE="${ROOT}${DOMAIN}_devtest_dials.json"

if [ "$DOMAIN" == "furniture" ]; then
    METADATA_FILE="${ROOT}furniture_metadata.csv"
elif [ "$DOMAIN" == "fashion" ]; then
    METADATA_FILE="${ROOT}fashion_metadata.json"
else
    echo "Invalid domain!"
    exit 0
fi

# Output files.
VOCAB_FILE="${ROOT}${DOMAIN}_vocabulary.json"
METADATA_EMBEDS="${ROOT}${DOMAIN}_asset_embeds.npy"
ATTR_VOCAB_FILE="${ROOT}${DOMAIN}_attribute_vocabulary.json"
MODEL_METAINFO="models/${DOMAIN}_model_metainfo.json"

# Train all models on a domain Save checkpoints and logs with unique label.
CUR_TIME=$(date +"_%m%d_%H:%M")
UNIQ_LABEL="${DETAILS}${CUR_TIME}"
CHECKPOINT_PATH="outputs/${UNIQ_LABEL}/checkpoints"
LOG_PATH="outputs/${UNIQ_LABEL}/logs"
TENSORBOARD_PATH="outputs/${UNIQ_LABEL}/runs"

mkdir -p outputs
mkdir -p outputs/${UNIQ_LABEL}
mkdir -p ${CHECKPOINT_PATH}
mkdir -p ${LOG_PATH}


COMMON_FLAGS="
    --train_data_path=${TRAIN_JSON_FILE/.json/_mm_inputs.npy} \
    --eval_data_path=${DEV_JSON_FILE/.json/_mm_inputs.npy} \
    --asset_embed_path=${METADATA_EMBEDS} \
    --metainfo_path=${MODEL_METAINFO} \
    --attr_vocab_path=${ATTR_VOCAB_FILE} \
    --learning_rate=0.0001 --gpu_id=$GPU_ID --use_action_attention \
    --num_epochs=100 --eval_every_epoch=1 --batch_size=20 \
    --save_every_epoch=1 --word_embed_size=256 --num_layers=2 \
    --hidden_size=512 --gate_type MAG\
    --use_multimodal_state --use_action_output --use_bahdanau_attention \
    --domain=${DOMAIN} --save_prudently --tensorboard_path=${TENSORBOARD_PATH}"


# Train history-agnostic model.
# For other models, please look at scripts/train_all_simmc_models.sh
python -u train_simmc_agent.py $COMMON_FLAGS \
     --encoder="history_agnostic" \
     --text_encoder="lstm" \
     --snapshot_path="${CHECKPOINT_PATH}/" &> "${LOG_PATH}/output.log" &

# Transformer model.
#python -u train_simmc_agent.py $COMMON_FLAGS \
#    --encoder="history_agnostic" \
#    --text_encoder="transformer" \
#    --num_heads_transformer=4 --num_layers_transformer=4 \
#    --hidden_size_transformer=2048 --hidden_size=256\
#    --snapshot_path="${CHECKPOINT_PATH}/transf/" &> "${LOG_PATH}/transf.log" &

# Evaluate a trained model checkpoint.
# CHECKPOINT_PATH="checkpoints/hae/epoch_20.tar"
# python -u eval_simmc_agent.py \
#     --eval_data_path=${DEV_JSON_FILE/.json/_mm_inputs.npy} \
#     --checkpoint="$CHECKPOINT_PATH" --gpu_id=0 --batch_size=50 \
#     --domain="$DOMAIN"
