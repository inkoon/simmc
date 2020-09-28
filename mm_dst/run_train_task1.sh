#!/bin/bash

# HYPER-PARAMETERS 	# OPTIONS

TRAIN_DATA=total	# { total | train }
MODEL_TYPE=gpt2		# { gpt2 | gpt2-medium | gpt2-large }
EPOCH=10
TRAIN_BATCH=8
VAL_BATCH=32
WARMUP=4000
LR=5e-5

if [[ $# -eq 2 ]]
then
	KEYWORD=$1
	GPU_ID=$2
else
	echo "ERROR : run format > ./run_train_gpt2.sh [KEYWORD] [GPU_ID]"
	exit 1
fi

PATH_DIR=$(realpath .)

# Train (furniture, multi-modal)
python -m gpt2_dst.scripts.run_language_modeling \
    --output_dir="${PATH_DIR}"/gpt2_dst/save/task1/furniture/"${KEYWORD}" \
    --model_type="${MODEL_TYPE}" \
    --model_name_or_path="${MODEL_TYPE}" \
    --line_by_line \
    --add_special_tokens="${PATH_DIR}"/gpt2_dst/data/task1/furniture/special_tokens.json \
    --do_train \
    --train_data_file="${PATH_DIR}"/gpt2_dst/data/task1/furniture/furniture_"${TRAIN_DATA}"_dials_target.txt \
    --logging_steps=0 \
    --save_steps=0 \
    --num_train_epochs=$EPOCH \
    --overwrite_output_dir \
    --learning_rate=$LR \
    --warmup_steps=$WARMUP \
    --gpu_id=$GPU_ID \
    --per_gpu_train_batch_size=$TRAIN_BATCH \
    --per_gpu_eval_batch_size=$VAL_BATCH

echo "Train Complete! Model saved in gpt2_dst/save/task1/furniture/$KEYWORD"

# Train (Fashion, multi-modal)
python -m gpt2_dst.scripts.run_language_modeling \
    --output_dir="${PATH_DIR}"/gpt2_dst/save/task1/fashion/"${KEYWORD}" \
    --model_type="${MODEL_TYPE}" \
    --model_name_or_path="${MODEL_TYPE}" \
    --line_by_line \
    --add_special_tokens="${PATH_DIR}"/gpt2_dst/data/task1/fashion/special_tokens.json \
    --do_train \
    --train_data_file="${PATH_DIR}"/gpt2_dst/data/task1/fashion/fashion_"${TRAIN_DATA}"_dials_target.txt \
    --logging_steps=0 \
    --save_steps=0 \
    --num_train_epochs=$EPOCH \
    --overwrite_output_dir \
    --learning_rate=$LR \
    --warmup_steps=$WARMUP \
    --gpu_id=$GPU_ID \
    --per_gpu_train_batch_size=$TRAIN_BATCH \
    --per_gpu_eval_batch_size=$VAL_BATCH

echo "Train Complete! Model saved in gpt2_dst/save/task1/fashion/$KEYWORD"
