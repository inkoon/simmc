#!/bin/bash

TEST_DATA=devtest		# {devtest|test}
PATH_DIR=$(realpath .)

if [[ $# -eq 2 ]]
then
	KEYWORD=$1
	GPU_ID=$2
else
	echo "error : run format > ./run_generate_gpt2.sh [KEYWORD] [GPU_ID]"
	exit 1
fi

# IMPORTANT : install normal transformers version 2.8.0

# Furniture
# Multimodal Data
CUDA_VISIBLE_DEVICES=$GPU_ID python -m gpt2_dst.scripts.run_generation \
    --model_type=gpt2 \
    --model_name_or_path="${PATH_DIR}"/gpt2_dst/save/task1/furniture/"${KEYWORD}" \
    --num_return_sequences=1 \
    --length=100 \
    --stop_token="<EOS>" \
    --num_beams=2 \
    --num_gen=10 \
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/task1/furniture/furniture_"${TEST_DATA}"_dials_predict.txt \
    --path_output="${PATH_DIR}"/gpt2_dst/results/task1/furniture/"${KEYWORD}"/furniture_"${TEST_DATA}"_dials_predicted.txt

# Fashion
# Multimodal Data
CUDA_VISIBLE_DEVICES=$GPU_ID python -m gpt2_dst.scripts.run_generation \
    --model_type=gpt2 \
    --model_name_or_path="${PATH_DIR}"/gpt2_dst/save/task1/fashion/"${KEYWORD}" \
    --num_return_sequences=1 \
    --length=100 \
    --stop_token="<EOS>" \
    --num_beams=2 \
    --num_gen=10 \
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/task1/fashion/fashion_"${TEST_DATA}"_dials_predict.txt \
    --path_output="${PATH_DIR}"/gpt2_dst/results/task1/fashion/"${KEYWORD}"/fashion_"${TEST_DATA}"_dials_predicted.txt
