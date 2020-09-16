#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
else
    PATH_DIR=$(realpath "$1")
fi

# Generate sentences (Furniture, text-only)
python -m gpt2_dst.scripts.run_generation \
    --model_type=gpt2 \
    --model_name_or_path="${PATH_DIR}"/gpt2_dst/save/furniture_to/ \
    --num_return_sequences=1 \
    --length=100 \
    --stop_token='<EOS>' \
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/toy_furniture_to/furniture_devtest_dials_predict.txt \
    --path_output="${PATH_DIR}"/gpt2_dst/results/toy_furniture_to/furniture_devtest_dials_predicted.txt

# Generate sentences (Furniture, multi-modal)
python -m gpt2_dst.scripts.run_generation \
    --model_type=gpt2 \
    --model_name_or_path="${PATH_DIR}"/gpt2_dst/save/furniture/ \
    --num_return_sequences=1 \
    --length=100 \
    --stop_token='<EOS>' \
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/toy_furniture/furniture_devtest_dials_predict.txt \
    --path_output="${PATH_DIR}"/gpt2_dst/results/toy_furniture/furniture_devtest_dials_predicted.txt

# Generate sentences (Fashion, text-only)
python -m gpt2_dst.scripts.run_generation \
    --model_type=gpt2 \
    --model_name_or_path="${PATH_DIR}"/gpt2_dst/save/fashion_to/ \
    --num_return_sequences=1 \
    --length=100 \
    --stop_token='<EOS>' \
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/toy_fashion_to/fashion_devtest_dials_predict.txt \
    --path_output="${PATH_DIR}"/gpt2_dst/results/toy_fashion_to/fashion_devtest_dials_predicted.txt

# Generate sentences (Fashion, multi-modal)
python -m gpt2_dst.scripts.run_generation \
    --model_type=gpt2 \
    --model_name_or_path="${PATH_DIR}"/gpt2_dst/save/fashion/ \
    --num_return_sequences=1 \
    --length=100 \
    --stop_token='<EOS>' \
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/toy_fashion/fashion_devtest_dials_predict.txt \
    --path_output="${PATH_DIR}"/gpt2_dst/results/toy_fashion/fashion_devtest_dials_predicted.txt
