#!/bin/bash

TEST_DATA=devtest		# {devtest|test}
PATH_DIR=$(realpath .)
PATH_DATA_DIR=$(realpath ..)

# Step 1 : Generate output for Task1 evaluation

# Furniture
# Multimodal Data
python -m gpt2_dst.scripts.task1_output \
    --input_predicted_file="${PATH_DIR}"/gpt2_dst/results/task1/furniture/ensemble/furniture_${TEST_DATA}_dials_predicted.txt \
    --output_path="${PATH_DIR}"/gpt2_dst/results/task1/furniture/ensemble/dstc9-simmc-teststd-furniture-subtask-1.json \
    --domain=furniture \
    --input_predict_json="${PATH_DATA_DIR}"/data/simmc_furniture/furniture_"${TEST_DATA}"_dials.json \
    

# Fashion
# Multimodal Data 
python -m gpt2_dst.scripts.task1_output \
    --input_predicted_file="${PATH_DIR}"/gpt2_dst/results/task1/fashion/ensemble/fashion_${TEST_DATA}_dials_predicted.txt \
    --output_path="${PATH_DIR}"/gpt2_dst/results/task1/fashion/ensemble/dstc9-simmc-teststd-fashion-subtask-1.json\
    --domain=fashion \
    --input_predict_json="${PATH_DATA_DIR}"/data/simmc_fashion/fashion_"${TEST_DATA}"_dials.json \
    