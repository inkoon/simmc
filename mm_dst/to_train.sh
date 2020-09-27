#!/bin/bash
if [[ $# -eq 0 ]] || [[ $# -eq 1 ]] || [[ $# -eq 2 ]]
then
	echo "run format > ./run_domain_keyword.sh [domain] [keyword]"
	exit 1

elif [[ $# -eq 3 ]]
then
	DOMAIN=$1
	KEYWORD=$2
	GPU_ID=$3
fi

PATH_DIR=$(realpath .)

# Train ("${DOMAIN}", multi-modal)
python -m gpt2_dst.scripts.run_language_modeling \
    --output_dir="${PATH_DIR}"/gpt2_dst/save/"${DOMAIN}"_to/"${KEYWORD}""${VERSION}" \
    --model_type=gpt2 \
    --model_name_or_path="${PATH_DIR}"/gpt2_dst/save/fine_tune \
    --line_by_line \
    --add_special_tokens="${PATH_DIR}"/gpt2_dst/data/"${DOMAIN}"_to/special_tokens.json \
    --do_train \
    --train_data_file="${PATH_DIR}"/gpt2_dst/data/"${DOMAIN}"_to/"${DOMAIN}"_train_dials_target.txt \
    --num_train_epochs=10 \
    --overwrite_output_dir \
    --gpu_id=$GPU_ID \
    --per_gpu_train_batch_size=8 \
    --per_gpu_eval_batch_size=32 \
    --warmup_steps=4000 \
    --logging_steps=0 \
    --save_steps=0 \
    --fp16 

echo "Train Complete! Text Only Model saved in save/$DOMAIN/$KEYWORD$VERSION"
