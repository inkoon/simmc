#!/bin/bash
if [[ $# -eq 0 ]] || [[ $# -eq 1 ]]
then
	echo "run format > ./run_domain_keyword.sh [domain] [keyword]"
	exit 1
elif [[ $# -eq 2 ]]
then
	DOMAIN=$1
	KEYWORD=$2
	VERSION=""
elif [[ $# -eq 3 ]]
then
	DOMAIN=$1
	KEYWORD=$2
	VERSION=$3
fi

GPU_ID=0

PATH_DIR=$(realpath .)

# Train ("${DOMAIN}", multi-modal)
python -m gpt2_dst.scripts.run_language_modeling \
    --output_dir="${PATH_DIR}"/gpt2_dst/save/"${DOMAIN}"/"${KEYWORD}""${VERSION}" \
    --model_type=gpt2-large \
    --model_name_or_path=gpt2-large \
    --line_by_line \
    --add_special_tokens="${PATH_DIR}"/gpt2_dst/data/"${DOMAIN}"/special_tokens.json \
    --do_train \
    --train_data_file="${PATH_DIR}"/gpt2_dst/data/"${DOMAIN}"/"${DOMAIN}"_total_dials_target.txt \
    --num_train_epochs=5 \
    --overwrite_output_dir \
    --gpu_id=$GPU_ID \
    --per_gpu_train_batch_size=2 \
    --per_gpu_eval_batch_size=8 \
    --warmup_steps=16000 \
    --logging_steps=0 \
    --save_steps=0 \
    --fp16 

echo "Train Complete! Model saved in save/$DOMAIN/$KEYWORD$VERSION"
