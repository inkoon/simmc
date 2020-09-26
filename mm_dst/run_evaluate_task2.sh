#!/bin/bash
if [[ $# -eq 2 ]]
then
	DOMAIN=$1
	KEYWORD=$2
else
	echo "run format > ./run_domain_keyword.sh [DOMAIN] [KEYWORD]"
	exit 1
fi

PATH_DIR=$(realpath .)

# Evaluate ("${DOMAIN}, multi-modal)
python -m gpt2_dst.scripts.evaluate_task2 \
    --input_path_target="${PATH_DIR}"/../data/simmc_"${DOMAIN}"/ \
    --input_path_predicted="${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"/"${KEYWORD}""${VERSION}"/"${DOMAIN}"_devtest_dials_predicted.txt \
    --output_path_report="${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"/"${KEYWORD}""${VERSION}"/"${DOMAIN}"_devtest_dials_report.json

