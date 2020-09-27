#!/bin/bash
if [[ $# -eq 1 ]]
then
	KEYWORD=$1
else
	echo "run format > ./ [KEYWORD]"
	exit 1
fi

PATH_DIR=$(realpath .)

python -m gpt2_dst.utils.make_task1_input \
    --path="${PATH_DIR}"/gpt2_dst/results/furniture/"${KEYWORD}"/ \
    --domain=furniture

python -m gpt2_dst.utils.make_task1_input \
    --path="${PATH_DIR}"/gpt2_dst/results/fashion/"${KEYWORD}"/ \
    --domain=fashion


echo "Processing task3 output for task1 input completed!"

cp "${PATH_DIR}"/gpt2_dst/results/furniture/"${KEYWORD}"/furniture_devtest_belief_state.json "${PATH_DIR}"/../data/simmc_furniture/

cp "${PATH_DIR}"/gpt2_dst/results/fashion/"${KEYWORD}"/fashion_devtest_belief_state.json "${PATH_DIR}"/../data/simmc_fashion/

echo "Input for running Task 1 moved to simmc/data"

