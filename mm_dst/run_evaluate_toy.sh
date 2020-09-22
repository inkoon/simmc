#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
else
    PATH_DIR=$(realpath "$1")
fi

# Evaluate (furniture, non-multimodal)
python -m gpt2_dst.scripts.evaluate \
    --input_path_target="${PATH_DIR}"/gpt2_dst/data/toy_furniture_to/furniture_devtest_dials_target.txt \
    --input_path_predicted="${PATH_DIR}"/gpt2_dst/results/toy_furniture_to/furniture_devtest_dials_predicted.txt \
    --output_path_report="${PATH_DIR}"/gpt2_dst/results/toy_furniture_to/furniture_devtest_dials_report.json

# Evaluate (furniture, multi-modal)
python -m gpt2_dst.scripts.evaluate \
    --input_path_target="${PATH_DIR}"/gpt2_dst/data/toy_furniture/furniture_devtest_dials_target.txt \
    --input_path_predicted="${PATH_DIR}"/gpt2_dst/results/toy_furniture/furniture_devtest_dials_predicted.txt \
    --output_path_report="${PATH_DIR}"/gpt2_dst/results/toy_furniture/furniture_devtest_dials_report.json

# Evaluate (Fashion, non-multimodal)
python -m gpt2_dst.scripts.evaluate \
    --input_path_target="${PATH_DIR}"/gpt2_dst/data/toy_fashion_to/fashion_devtest_dials_target.txt \
    --input_path_predicted="${PATH_DIR}"/gpt2_dst/results/toy_fashion_to/fashion_devtest_dials_predicted.txt \
    --output_path_report="${PATH_DIR}"/gpt2_dst/results/toy_fashion_to/fashion_devtest_dials_report.json

# Evaluate (Fashion, multi-modal)
python -m gpt2_dst.scripts.evaluate \
    --input_path_target="${PATH_DIR}"/gpt2_dst/data/toy_fashion/fashion_devtest_dials_target.txt \
    --input_path_predicted="${PATH_DIR}"/gpt2_dst/results/toy_fashion/fashion_devtest_dials_predicted.txt \
    --output_path_report="${PATH_DIR}"/gpt2_dst/results/toy_fashion/fashion_devtest_dials_report.json
