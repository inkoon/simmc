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

PATH_DIR=$(realpath .)

# Evaluate ("${DOMAIN}, multi-modal)
python -m gpt2_dst.scripts.evaluate \
    --input_path_target="${PATH_DIR}"/gpt2_dst/data/"${DOMAIN}"/"${DOMAIN}"_devtest_dials_target.txt \
    --input_path_predicted="${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"/"${KEYWORD}""${VERSION}"/"${DOMAIN}"_devtest_dials_predicted.txt \
    --output_path_report="${PATH_DIR}"/gpt2_dst/results/"${DOMAIN}"/"${KEYWORD}""${VERSION}"/"${DOMAIN}"_devtest_dials_report.json
