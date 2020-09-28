#!/bin/bash
PATH_DIR=$(realpath .)

# HYPER-PARAMETERS 

TEST_DATA=test 	# {devtest|test}

if [[ $# -eq 1 ]]
then
	GPU_ID=$1
else
	echo "error : run format > ./run_generate_gpt2.sh [GPU_ID]"
	exit 1
fi

# Furniture
# Multimodal Data
CUDA_VISIBLE_DEVICES=$GPU_ID python -m gpt2_dst.scripts.run_generation_en \
    --model_type=gpt2 \
    --model_name_or_path_list "${PATH_DIR}"/gpt2_dst/save/task1/furniture/td_large "${PATH_DIR}"/gpt2_dst/save/task1/furniture/large \
    --num_return_sequences=1 \
    --length=100 \
    --stop_token="<EOS>" \
    --num_beams=2 \
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/task1/furniture/furniture_"${TEST_DATA}"_dials_predict.txt \
    --path_output="${PATH_DIR}"/gpt2_dst/results/task1/furniture/ensemble/furniture_"${TEST_DATA}"_dials_predicted.txt

# Fashion
# Multimodal Data
CUDA_VISIBLE_DEVICES=$GPU_ID python -m gpt2_dst.scripts.run_generation_en \
    --model_type=gpt2 \
    --model_name_or_path "${PATH_DIR}"/gpt2_dst/save/task1/fashion/td_large "${PATH_DIR}"/gpt2_dst/save/fashion/td_small \
    --num_return_sequences=1 \
    --length=100 \
    --stop_token="<EOS>" \
    --num_beams=2 \
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/task1/fashion/fashion_"${TEST_DATA}"_dials_predict.txt \
    --path_output="${PATH_DIR}"/gpt2_dst/results/task1/fashion/ensemble/fashion_"${TEST_DATA}"_dials_predicted.txt

