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

PATH_DIR=$(realpath .)

# Generate sentences ("${DOMAIN}", multi-modal)
CUDA_VISIBLE_DEVICES=$GPU_ID python -m gpt2_dst.scripts.run_generation \
    --model_type=gpt2 \
    --model_name_or_path="${PATH_DIR}"/gpt2_dst/save/"${DOMAIN}"_to/"${KEYWORD}""${VERSION}" \
    --num_return_sequences=1 \
    --length=100 \
    --gpu_id=$GPU_ID \
    --stop_token="<EOS>" \
    --num_beams=2 \
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/"${DOMAIN}"_to/"${DOMAIN}"_devtest_dials_predict.txt \
    --path_output="${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"_to/"${KEYWORD}""${VERSION}"/"${DOMAIN}"_devtest_dials_predicted.txt

python -m gpt2_dst.utils.total_postprocess \
    --path="${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"_to/"${KEYWORD}""${VERSION}"/ \
    --domain="${DOMAIN}"


mv "${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"_to/"${KEYWORD}""${VERSION}"/"${DOMAIN}"_devtest_dials_predicted.txt "${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"_to/"${KEYWORD}""${VERSION}"/"${DOMAIN}"_devtest_dials_predicted.org

mv "${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"_to/"${KEYWORD}""${VERSION}"/"${DOMAIN}"_devtest_dials_predicted_processed.txt "${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"_to/"${KEYWORD}""${VERSION}"/"${DOMAIN}"_devtest_dials_predicted.txt
