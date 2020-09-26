#!/bin/bash
if [[ $# -eq 3 ]]
then
	DOMAIN=$1
	KEYWORD=$2
	GPU_ID=$3
else
	echo "run format > ./run_domain_keyword.sh [furniture/fashion] [KEYWORD] [GPU_ID]"
	exit 1
fi

PATH_DIR=$(realpath .)

# PARAMTERS
TRAIN_DATA=train
MODEL_TYPE=gpt2
EPOCH=15
BATCH=16
VAL_BATCH=64
WARMUP=4000


PARAMS="train_data : $TRAIN_DATA
model_type : $MODEL_TYPE
num_train_epochs : $EPOCH
per_gpu_train_batch_size : $BATCH
warmup_steps : $WARMUP"

# Train ("${DOMAIN}", multi-modal)
python -m gpt2_dst.scripts.run_language_modeling \
    --output_dir="${PATH_DIR}"/gpt2_dst/save/"${DOMAIN}"_to/"${KEYWORD}" \
    --model_type="${MODEL_TYPE}" \
    --model_name_or_path="${PATH_DIR}"/gpt2_dst/save/fine_tune \
    --line_by_line \
    --add_special_tokens="${PATH_DIR}"/gpt2_dst/data/"${DOMAIN}"_to/special_tokens.json \
    --do_train \
    --train_data_file="${PATH_DIR}"/gpt2_dst/data/"${DOMAIN}"_to/"${DOMAIN}"_"${TRAIN_DATA}"_dials_target.txt \
    --num_train_epochs=$EPOCH \
    --overwrite_output_dir \
    --gpu_id=$GPU_ID \
    --per_gpu_train_batch_size=$BATCH \
    --per_gpu_eval_batch_size=$VAL_BATCH \
    --warmup_steps=$WARMUP \
    --logging_steps=0 \
    --save_steps=0 \
    --fp16 

echo "Train Complete! Model saved in save/$DOMAIN/$KEYWORD$VERSION"

# Generate sentences ("${DOMAIN}", multi-modal)
CUDA_VISIBLE_DEVICES=$GPU_ID python -m gpt2_dst.scripts.run_generation \
    --model_type=gpt2 \
    --model_name_or_path="${PATH_DIR}"/gpt2_dst/save/"${DOMAIN}"_to/"${KEYWORD}" \
    --num_return_sequences=1 \
    --length=100 \
    --gpu_id=$GPU_ID \
    --stop_token="<EOS>" \
    --num_beams=2 \
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/"${DOMAIN}"_to/"${DOMAIN}"_devtest_dials_predict.txt \
    --path_output="${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"_to/"${KEYWORD}"/"${DOMAIN}"_devtest_dials_predicted.txt

python -m gpt2_dst.utils.total_postprocess \
    --path="${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"_to/"${KEYWORD}"/ \
    --domain="${DOMAIN}"

echo "$PARAMS" > "${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"_to/"${KEYWORD}"/parameters.txt

mv "${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"_to/"${KEYWORD}"/"${DOMAIN}"_devtest_dials_predicted.txt "${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"_to/"${KEYWORD}"/"${DOMAIN}"_devtest_dials_predicted.org

mv "${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"_to/"${KEYWORD}"/"${DOMAIN}"_devtest_dials_predicted_processed.txt "${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"_to/"${KEYWORD}"/"${DOMAIN}"_devtest_dials_predicted.txt

# Evaluate ("${DOMAIN}, multi-modal)
python -m gpt2_dst.scripts.evaluate \
    --input_path_target="${PATH_DIR}"/gpt2_dst/data/"${DOMAIN}"_to/"${DOMAIN}"_devtest_dials_target.txt \
    --input_path_predicted="${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"_to/"${KEYWORD}"/"${DOMAIN}"_devtest_dials_predicted.txt \
    --output_path_report="${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"_to/"${KEYWORD}"/"${DOMAIN}"_devtest_dials_report.json
