#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
    PATH_DATA_DIR=$(realpath ../data)
else
    PATH_DIR=$(realpath "$1")
    PATH_DATA_DIR=$(realpath "$2")
fi
'
# Fashion
# Multimodal Data
# Train split
python -m gpt2_dst.scripts.preprocess_input \
    --input_path_json="${PATH_DATA_DIR}"/simmc_fashion/fashion_train_dials.json \
    --output_path_predict="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_train_dials_predict.txt \
    --output_path_target="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_train_dials_target.txt \
    --len_context=2 \
    --use_multimodal_contexts=1 \
    --output_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/fashion/special_tokens.json

# Dev split
python -m gpt2_dst.scripts.preprocess_input \
    --input_path_json="${PATH_DATA_DIR}"/simmc_fashion/fashion_dev_dials.json \
    --output_path_predict="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_dev_dials_predict.txt \
    --output_path_target="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_dev_dials_target.txt \
    --len_context=2 \
    --use_multimodal_contexts=1 \
    --input_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/fashion/special_tokens.json \
    --output_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/fashion/special_tokens.json \

# Devtest split
python -m gpt2_dst.scripts.preprocess_input \
    --input_path_json="${PATH_DATA_DIR}"/simmc_fashion/fashion_devtest_dials.json \
    --output_path_predict="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_devtest_dials_predict.txt \
    --output_path_target="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_devtest_dials_target.txt \
    --len_context=2 \
    --use_multimodal_contexts=1 \
    --input_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/fashion/special_tokens.json \

# Test split
# python -m gpt2_dst.scripts.preprocess_input \
#    --input_path_json="${PATH_DATA_DIR}"/simmc_fashion/fashion_test_dials.json \
#    --output_path_predict="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_test_dials_predict.txt \
#    --output_path_target="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_test_dials_target.txt \
#    --len_context=2 \
#    --use_multimodal_contexts=1 \
#    --input_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/fashion/special_tokens.json \
'
# Furniture
# Multimodal Data
# Train split
python -m gpt2_dst.scripts.preprocess_input \
    --input_path_json="${PATH_DATA_DIR}"/simmc_furniture/furniture_train_dials.json \
    --attribute_path_json="${PATH_DATA_DIR}"/simmc_furniture/furniture_attribute/furniture_train_attribute_list.json \
    --attribute_vocab_json="${PATH_DATA_DIR}"/simmc_furniture/furniture_attribute/furniture_attribute_vocabulary.json \
    --output_path_predict="${PATH_DIR}"/gpt2_dst/data/task1/furniture/furniture_train_dials_predict.txt \
    --output_path_target="${PATH_DIR}"/gpt2_dst/data/task1/furniture/furniture_train_dials_target.txt \
    --len_context=2 \
    --use_multimodal_contexts=1 \
    --task1 \
    --output_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/furniture/special_tokens.json
'
# Dev split
python -m gpt2_dst.scripts.preprocess_input \
    --input_path_json="${PATH_DATA_DIR}"/simmc_furniture/furniture_dev_dials.json \
    --attribute_path_json="${PATH_DATA_DIR}"/simmc_furniture/furniture_attribute/furniture_dev_attribute_list.json \
    --attribute_vocab_json="${PATH_DATA_DIR}"/simmc_furniture/furniture_attribute/furniture_attribute_vocab.json \
    --output_path_predict="${PATH_DIR}"/gpt2_dst/data/task1/furniture/furniture_dev_dials_predict.txt \
    --output_path_target="${PATH_DIR}"/gpt2_dst/data/task1/furniture/furniture_dev_dials_target.txt \
    --len_context=2 \
    --use_multimodal_contexts=1 \
    --task1 \
    --input_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/furniture/special_tokens.json \
    --output_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/furniture/special_tokens.json \

# Devtest split
python -m gpt2_dst.scripts.preprocess_input \
    --input_path_json="${PATH_DATA_DIR}"/simmc_furniture/furniture_devtest_dials.json \
    --attribute_path_json="${PATH_DATA_DIR}"/simmc_furniture/furniture_attribute/furniture_devtest_attribute_list.json \
    --attribute_vocab_json="${PATH_DATA_DIR}"/simmc_furniture/furniture_attribute/furniture_attribute_vocab.json \
    --output_path_predict="${PATH_DIR}"/gpt2_dst/data/task1/furniture/furniture_devtest_dials_predict.txt \
    --output_path_target="${PATH_DIR}"/gpt2_dst/data/task1/furniture/furniture_devtest_dials_target.txt \
    --len_context=2 \
    --use_multimodal_contexts=1 \
    --task1 \
    --input_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/furniture/special_tokens.json \
'
# Test split
#python -m gpt2_dst.scripts.preprocess_input \
#    --input_path_json="${PATH_DATA_DIR}"/simmc_furniture/furniture_test_dials.json \
#    --output_path_predict="${PATH_DIR}"/gpt2_dst/data/furniture/furniture_test_dials_predict.txt \
#    --output_path_target="${PATH_DIR}"/gpt2_dst/data/furniture/furniture_test_dials_target.txt \
#    --len_context=2 \
#    --use_multimodal_contexts=1 \
#    --input_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/furniture/special_tokens.json \
