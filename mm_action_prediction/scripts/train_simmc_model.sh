#!/bin/bash

# Check the parameters
if [ $# -lt 2 ];then
    echo "There's no parameters -> ./run.sh [model descriptions] [gpu no]"
    exit 1
fi

DOMAIN="furniture"
# DOMAIN="fashion"
ROOT="../data/simmc_${DOMAIN}/"
DETAILS=$1
GPU_ID=$2


# Input files.
TRAIN_JSON_FILE="${ROOT}${DOMAIN}_train_dials.json"
TRAINDEV_JSON_FILE="${ROOT}${DOMAIN}_traindev_dials.json"
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
    --train_data_path=${TRAINDEV_JSON_FILE/.json/_mm_inputs.npy} \
    --eval_data_path=${DEVTEST_JSON_FILE/.json/_mm_inputs.npy} \
    --asset_embed_path=${METADATA_EMBEDS} \
    --metainfo_path=${MODEL_METAINFO} \
    --attr_vocab_path=${ATTR_VOCAB_FILE} \
    --learning_rate=0.0002 --gpu_id=$GPU_ID --use_action_attention \
    --num_epochs=60 --eval_every_epoch=4 --batch_size=32 \
    --save_every_epoch=4 --word_embed_size=300 --num_layers=2 \
    --hidden_size=512 \
    --use_multimodal_state --use_action_output --use_bahdanau_attention \
    --domain=${DOMAIN} --save_prudently --tensorboard_path=${TENSORBOARD_PATH}"


# History-agnostic model.
# python -u train_simmc_agent.py $COMMON_FLAGS \
#      --encoder="history_agnostic" --text_encoder="lstm" --embedding_type="glove" --gate_type="MAG"\
#      --snapshot_path="${CHECKPOINT_PATH}/" &> "${LOG_PATH}/hae.log" &

# Hierarchical recurrent encoder model.
# python -u train_simmc_agent.py $COMMON_FLAGS \
#     --encoder="hierarchical_recurrent" --text_encoder="lstm" --embedding_type="random" --gate_type="MAG" \
#     --snapshot_path="${CHECKPOINT_PATH}/" &> "${LOG_PATH}/hre.log" &

# Memory encoder model.
python -u train_simmc_agent.py $COMMON_FLAGS \
    --encoder="memory_network" --text_encoder="lstm" --embedding_type="random" --gate_type="MAG" \
    --snapshot_path="${CHECKPOINT_PATH}/" &> "${LOG_PATH}/mn.log" &

# # TF-IDF model.
# python -u train_simmc_agent.py $COMMON_FLAGS \
#     --encoder="tf_idf" --text_encoder="lstm" \
#     --snapshot_path="${CHECKPOINT_PATH}/" &> "${LOG_PATH}/tf_idf.log" &
#
# # Transformer model.
# python -u train_simmc_agent.py $COMMON_FLAGS \
#     --encoder="history_agnostic" \
#     --text_encoder="transformer" \
#     --num_heads_transformer=4 --num_layers_transformer=4 \
#     --hidden_size_transformer=2048 --hidden_size=256\
#     --snapshot_path="${CHECKPOINT_PATH}/" &> "${LOG_PATH}/transf.log" &

# Evaluate a trained model checkpoint.
# CHECKPOINT_PATH="checkpoints/hae/epoch_20.tar"
# python -u eval_simmc_agent.py \
#     --eval_data_path=${DEV_JSON_FILE/.json/_mm_inputs.npy} \
#     --checkpoint="$CHECKPOINT_PATH" --gpu_id=0 --batch_size=50 \
#     --domain="$DOMAIN"
