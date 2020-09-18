#!/bin/bash
PATH_DIR=$(realpath .)
PATH_DATA_DIR=$(realpath ../data)


# Fashion
# Multimodal Data
# Devtest split
python -m gpt2_dst.scripts.preprocess_input \
    --input_path_json="${PATH_DATA_DIR}"/simmc_fashion/fashion_devtest_dials.json \
    --output_path_predict="${PATH_DIR}"/gpt2_dst/data/fashion/dummp_predict.txt \
    --output_path_target="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_devtest_dials_context.txt \
    --len_context=1 \
    --use_multimodal_contexts=1 \
    --input_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/fashion/special_tokens.json \

# Furniture
# Multimodal Data
# Devtest split
python -m gpt2_dst.scripts.preprocess_input \
    --input_path_json="${PATH_DATA_DIR}"/simmc_furniture/furniture_devtest_dials.json \
    --output_path_predict="${PATH_DIR}"/gpt2_dst/data/furniture/dummp_predict.txt \
    --output_path_target="${PATH_DIR}"/gpt2_dst/data/furniture/furniture_devtest_dials_context.txt \
    --len_context=1 \
    --use_multimodal_contexts=1 \
    --input_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/furniture/special_tokens.json \


