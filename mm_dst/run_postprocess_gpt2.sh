#!/bin/bash

TEST_DATA=test		# {devtest|test}
PATH_DIR=$(realpath .)
PATH_DATA_DIR=$(realpath ..)

# Step 1 : Generate output for Task3 evaluation

# Furniture
# Multimodal Data
python -m gpt2_dst.scripts.postprocess_output \
    --path="${PATH_DIR}"/gpt2_dst/results/furniture/ensemble/ \
    --domain=furniture \
    --data="${TEST_DATA}"

# Fashion
# Multimodal Data
python -m gpt2_dst.scripts.postprocess_output \
    --path="${PATH_DIR}"/gpt2_dst/results/fashion/ensemble/ \
    --domain=fashion \
    --data="${TEST_DATA}"

cp "${PATH_DIR}"/gpt2_dst/results/furniture/ensemble/furniture_"${TEST_DATA}"_belief_state.json "${PATH_DATA_DIR}"/data/belief_simmc_furniture/furniture_"${TEST_DATA}"_belief_state.json

cp "${PATH_DIR}"/gpt2_dst/results/fashion/ensemble/fashion_"${TEST_DATA}"_belief_state.json "${PATH_DATA_DIR}"/data/belief_simmc_fashion/fashion_"${TEST_DATA}"_belief_state.json

# Step 2 : Generate output for Task2 evaluation
python -m gpt2_dst.scripts.task2_output \
    --path="${PATH_DIR}"/gpt2_dst/results/furniture/ensemble/ \
    --domain=furniture \
    --data="${TEST_DATA}" \
    --dials_path="${PATH_DATA_DIR}"/data/simmc_furniture/furniture_"${TEST_DATA}"_dials.json \
    --retrieval_candidate_path="${PATH_DATA_DIR}"/data/simmc_furniture/furniture_"${TEST_DATA}"_dials_retrieval_candidates.json
# Fashion
python -m gpt2_dst.scripts.task2_output \
    --path="${PATH_DIR}"/gpt2_dst/results/fashion/ensemble/ \
    --domain=fashion \
    --data="${TEST_DATA}" \
    --dials_path="${PATH_DATA_DIR}"/data/simmc_fashion/fashion_"${TEST_DATA}"_dials.json \
    --retrieval_candidate_path="${PATH_DATA_DIR}"/data/simmc_fashion/fashion_"${TEST_DATA}"_dials_retrieval_candidates.json
