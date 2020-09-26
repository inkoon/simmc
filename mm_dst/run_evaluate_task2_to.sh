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
PATH_DATA_DIR=$(realpath ../data)

# Write response only data
python -m gpt2_dst.scripts.write_response \
    --path="${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"_to/"${KEYWORD}"/ \
    --domain=$DOMAIN

echo "response only file written in the results directory!"

# Evaluate ("${DOMAIN}, multi-modal)
python -m gpt2_dst.scripts.evaluate_task2 \
    --input_path_target="${PATH_DATA_DIR}"/simmc_"${DOMAIN}"/"${DOMAIN}"_devtest_dials.json \
    --retrieval_candidate_path="${PATH_DATA_DIR}"/simmc_"${DOMAIN}"/"${DOMAIN}"_devtest_dials_retrieval_candidates.json \
    --input_path_predicted="${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"_to/"${KEYWORD}"/"${DOMAIN}"_devtest_dials_predicted_response.txt \
    --output_path_report="${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"_to/"${KEYWORD}"/"${DOMAIN}"_devtest_dials_task2_report.json

