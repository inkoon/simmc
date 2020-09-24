#!/bin/bash
if [[ $# -eq 0 ]] || [[ $# -eq 1 ]]
then
	echo "run format > ./run_domain_keyword.sh [domain] [keyword]"
	exit 1
else
	DOMAIN=$1
	KEYWORD=$2
fi

GPU_ID='0'
NUM_GEN=500

PATH_DIR=$(realpath .)
PATH_DATA_DIR=$(realpath ../data)
'
# Generate sentences ("${DOMAIN}", multi-modal)
CUDA_VISIBLE_DEVICES=$GPU_ID python -m gpt2_dst.scripts.run_generation \
    --model_type=gpt2 \
    --model_name_or_path="${PATH_DIR}"/gpt2_dst/save/"${DOMAIN}"/"${KEYWORD}"\
    --num_return_sequences=1 \
    --length=100 \
    --gpu_id=$GPU_ID \
    --stop_token="<EOS>" \
    --num_beams=2 \
    --num_gen=$NUM_GEN \
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/toy_"${DOMAIN}"_"${KEYWORD}"/"${DOMAIN}"_devtest_dials_predict.txt \
    --path_output="${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"/toy_"${KEYWORD}"/"${DOMAIN}"_devtest_dials_predicted.txt

'
# Evaluate ("${DOMAIN}, multi-modal)
python -m gpt2_dst.scripts.evaluate \
    --input_path_target="${PATH_DIR}"/gpt2_dst/data/toy_"${DOMAIN}"/"${DOMAIN}"_devtest_dials_target.txt \
    --input_path_predicted="${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"/"${KEYWORD}"/"${DOMAIN}"_devtest_dials_predicted.txt \
    --output_path_report="${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"/"${KEYWORD}"/"${DOMAIN}"_devtest_dials_report.json


#--input_path_target="${PATH_DIR}"/gpt2_dst/data/toy_"${DOMAIN}"_"${KEYWORD}"/"${DOMAIN}"_devtest_dials_target.txt \


