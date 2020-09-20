#!/bin/bash

# PARAMETERS	DEFAUT VALUE
EPOCHS=10	# 1
TRAIN_BATCH=16	# 4
VAL_BATCH=32	# 4
MUL_GPU=0	# 1
GPU_ID='1'	# '0'
WARM_UP=2000	# 0
LR=1e-5		# 5e-5
LOGGING=2000	# 500
N_GPU=2

PARAMETERS="EPOCHS		$EPOCHS
TRAIN_BATCH	$TRAIN_BATCH
VAL_BATCH	$VAL_BATCH
MUL_GPU		$MUL_GPU
N_GPU 		$N_GPU
"

if [[ $# -eq 0 ]]
then
	echo "error : run format > ./*.sh (domain) keyword"
	exit 1
fi

if [[ $# -eq 1 ]]
then
	DOMAIN="all"
	KEYWORD=$1
fi

if [[ $# -eq 2 ]]
then
	DOMAIN=$1
	KEYWORD=$2
fi

PATH_DIR=$(realpath .)

if [ $DOMAIN == "furniture_to" ] || [ $DOMAIN == "all" ]
then
# Train (furniture, text-only)
python -m gpt2_dst.scripts.run_language_modeling \
    --output_dir="${PATH_DIR}"/gpt2_dst/save/furniture_to/"${KEYWORD}" \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --line_by_line \
    --add_special_tokens="${PATH_DIR}"/gpt2_dst/data/furniture_to/special_tokens.json \
    --do_train \
    --train_data_file="${PATH_DIR}"/gpt2_dst/data/furniture_to/furniture_train_dials_target.txt \
    --do_eval \
    --eval_data_file="${PATH_DIR}"/gpt2_dst/data/furniture_to/furniture_dev_dials_target.txt \
    --evaluate_during_training \
    --logging_steps=$LOGGING \
    --num_train_epochs=$EPOCHS \
    --overwrite_output_dir \
    --learning_rate=$LR \
    --warmup_steps=$WARM_UP \
    --gpu_id=$GPU_ID \
    --mul_gpu=$MUL_GPU \
    --n_gpu=$N_GPU \
    --per_gpu_train_batch_size=$TRAIN_BATCH \
    --per_gpu_eval_batch_size=$VAL_BATCH
fi

if [ $DOMAIN == "furniture" ] || [ $DOMAIN == "all" ]
then
# Train (furniture, multi-modal)
python -m gpt2_dst.scripts.run_language_modeling \
    --output_dir="${PATH_DIR}"/gpt2_dst/save/furniture/"${KEYWORD}" \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --line_by_line \
    --add_special_tokens="${PATH_DIR}"/gpt2_dst/data/furniture/special_tokens.json \
    --do_train \
    --train_data_file="${PATH_DIR}"/gpt2_dst/data/furniture/furniture_train_dials_target.txt \
    --do_eval \
    --eval_data_file="${PATH_DIR}"/gpt2_dst/data/furniture/furniture_dev_dials_target.txt \
    --evaluate_during_training \
    --logging_steps=$LOGGING \
    --num_train_epochs=$EPOCHS \
    --overwrite_output_dir \
    --learning_rate=$LR \
    --warmup_steps=$WARM_UP \
    --gpu_id=$GPU_ID \
    --mul_gpu=$MUL_GPU \
    --n_gpu=$N_GPU \
    --per_gpu_train_batch_size=$TRAIN_BATCH \
    --per_gpu_eval_batch_size=$VAL_BATCH
fi

if [ $DOMAIN == "fashion_to" ] || [ $DOMAIN == "all" ]
then
# Train (Fashion, text-only)
python -m gpt2_dst.scripts.run_language_modeling \
    --output_dir="${PATH_DIR}"/gpt2_dst/save/fashion_to/"${KEYWORD}" \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --line_by_line \
    --add_special_tokens="${PATH_DIR}"/gpt2_dst/data/fashion_to/special_tokens.json \
    --do_train \
    --train_data_file="${PATH_DIR}"/gpt2_dst/data/fashion_to/fashion_train_dials_target.txt \
    --do_eval \
    --eval_data_file="${PATH_DIR}"/gpt2_dst/data/fashion_to/fashion_dev_dials_target.txt \
    --evaluate_during_training \
    --logging_steps=$LOGGING \
    --num_train_epochs=$EPOCHS \
    --overwrite_output_dir \
    --learning_rate=$LR \
    --warmup_steps=$WARM_UP \
    --mul_gpu=$MUL_GPU \
    --gpu_id=$GPU_ID \
    --n_gpu=$N_GPU \
    --mul_gpu=$MUL_GPU \
    --per_gpu_train_batch_size=$TRAIN_BATCH \
    --per_gpu_eval_batch_size=$VAL_BATCH
fi

if [ $DOMAIN == "fashion" ] || [ $DOMAIN == "all" ]
then
# Train (Fashion, multi-modal)
python -m gpt2_dst.scripts.run_language_modeling \
    --output_dir="${PATH_DIR}"/gpt2_dst/save/fashion/"${KEYWORD}" \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --line_by_line \
    --add_special_tokens="${PATH_DIR}"/gpt2_dst/data/fashion/special_tokens.json \
    --do_train \
    --train_data_file="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_train_dials_target.txt \
    --do_eval \
    --eval_data_file="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_dev_dials_target.txt \
    --evaluate_during_training \
    --logging_steps=$LOGGING \
    --num_train_epochs=$EPOCHS \
    --overwrite_output_dir \
    --learning_rate=$LR \
    --warmup_steps=$WARM_UP \
    --mul_gpu=$MUL_GPU \
    --gpu_id=$GPU_ID \
    --n_gpu=$N_GPU \
    --mul_gpu=$MUL_GPU \
    --per_gpu_train_batch_size=$TRAIN_BATCH \
    --per_gpu_eval_batch_size=$VAL_BATCH
fi
