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

GPU_ID=1
#MUL_GPU=0


PATH_DIR=$(realpath .)

# "${DOMAIN}"
# Multimodal Data
# Train split

# Train ("${DOMAIN}", multi-modal)
python -m gpt2_dst.scripts.run_language_modeling \
    --output_dir="${PATH_DIR}"/gpt2_dst/save/"${DOMAIN}"/"${KEYWORD}""${VERSION}" \
    --model_type=gpt2-xl \
    --model_name_or_path=gpt2-xl \
    --line_by_line \
    --add_special_tokens="${PATH_DIR}"/gpt2_dst/data/"${DOMAIN}"/special_tokens.json \
    --do_train \
    --train_data_file="${PATH_DIR}"/gpt2_dst/data/"${DOMAIN}"/"${DOMAIN}"_train_dials_target.txt \
    --do_eval \
    --eval_data_file="${PATH_DIR}"/gpt2_dst/data/"${DOMAIN}"/"${DOMAIN}"_dev_dials_target.txt \
    --evaluate_during_training \
    --num_train_epochs=5 \
    --overwrite_output_dir \
    --gpu_id=$GPU_ID \
    --per_gpu_train_batch_size=8 \
    --per_gpu_eval_batch_size=32 \
    --warmup_steps=4000 \
    --fp16 \
    --logging_steps=0 \
    --save_steps=0

# Generate sentences ("${DOMAIN}", multi-modal)
CUDA_VISIBLE_DEVICES=$GPU_ID python -m gpt2_dst.scripts.run_generation \
    --model_type=gpt2 \
    --model_name_or_path="${PATH_DIR}"/gpt2_dst/save/"${DOMAIN}"/"${KEYWORD}""${VERSION}" \
    --num_return_sequences=1 \
    --length=100 \
    --gpu_id=$GPU_ID \
    --stop_token="<EOS>" \
    --num_beams=2 \
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/"${DOMAIN}"/"${DOMAIN}"_devtest_dials_predict.txt \
    --path_output="${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"/"${KEYWORD}""${VERSION}"/"${DOMAIN}"_devtest_dials_predicted.txt

python -m gpt2_dst.utils.total_postprocess \
    --path="${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"/"${KEYWORD}""${VERSION}"/"${DOMAIN}"_devtest_dials_predicted.txt \
    --domain="${DOMAIN}"

mv "${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"/"${KEYWORD}""${VERSION}"/"${DOMAIN}"_devtest_dials_predicted.txt "${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"/"${KEYWORD}""${VERSION}"/"${DOMAIN}"_devtest_dials_predicted.org

mv "${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"/"${KEYWORD}""${VERSION}"/"${DOMAIN}"_devtest_dials_predicted_processed.txt "${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"/"${KEYWORD}""${VERSION}"/"${DOMAIN}"_devtest_dials_predicted.txt

# Evaluate ("${DOMAIN}, multi-modal)
python -m gpt2_dst.scripts.evaluate \
    --input_path_target="${PATH_DIR}"/gpt2_dst/data/"${DOMAIN}"/"${DOMAIN}"_devtest_dials_target.txt \
    --input_path_predicted="${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"/"${KEYWORD}""${VERSION}"/"${DOMAIN}"_devtest_dials_predicted.txt \
    --output_path_report="${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"/"${KEYWORD}""${VERSION}"/"${DOMAIN}"_devtest_dials_report.json
