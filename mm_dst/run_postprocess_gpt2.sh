#!/bin/bash
if [[ $# -eq 0 ]] || [[ $# -eq 1 ]]
then
	echo "run format > ./run_postprocess_gpt2.sh [domain] [keyword]"
	exit 1
elif [[ $# -eq 2 ]]
then
	DOMAIN=$1
	KEYWORD=$2
fi

PATH_DIR=$(realpath .)


python -m gpt2_dst.utils.attribute_postprocess \
	--path="${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"/"${KEYWORD}"/

mv "${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"/"${KEYWORD}"/"${DOMAIN}"_devtest_dials_predicted.txt \
	"${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"/"${KEYWORD}"/"${DOMAIN}"_devtest_dials_predicted.org

mv "${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"/"${KEYWORD}"/"${DOMAIN}"_devtest_dials_predicted_processed.txt \
	"${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"/"${KEYWORD}"/"${DOMAIN}"_devtest_dials_predicted.txt

ls "${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"/"${KEYWORD}"

