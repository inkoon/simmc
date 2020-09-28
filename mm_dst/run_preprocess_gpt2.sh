#!/bin/bash
if [[ $# -lt 1 ]]
then
    PATH_DIR=$(realpath .)
    PATH_DATA_DIR=$(realpath ../data)
else
    PATH_DIR=$(realpath "$1")
    PATH_DATA_DIR=$(realpath "$2")
fi


# --------------------------------------------- #
#  STEP 1 : Get .txt dialog from .json data 	#
# ---------------------------------------------	#

echo "Step 1 : Generating (.txt) dialog from (.json) data..."

# Furniture
# Multimodal Data
# Train split
python -m gpt2_dst.scripts.preprocess_input \
    --input_path_json="${PATH_DATA_DIR}"/simmc_furniture/furniture_train_dials.json \
    --output_path_predict="${PATH_DIR}"/gpt2_dst/data/furniture/furniture_train_dials_predict.txt \
    --output_path_target="${PATH_DIR}"/gpt2_dst/data/furniture/furniture_train_dials_target.txt \
    --len_context=2 \
    --use_multimodal_contexts=1 \
    --output_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/furniture/special_tokens.json

# Dev split
python -m gpt2_dst.scripts.preprocess_input \
    --input_path_json="${PATH_DATA_DIR}"/simmc_furniture/furniture_dev_dials.json \
    --output_path_predict="${PATH_DIR}"/gpt2_dst/data/furniture/furniture_dev_dials_predict.txt \
    --output_path_target="${PATH_DIR}"/gpt2_dst/data/furniture/furniture_dev_dials_target.txt \
    --len_context=2 \
    --use_multimodal_contexts=1 \
    --input_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/furniture/special_tokens.json \
    --output_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/furniture/special_tokens.json \

# Devtest split
python -m gpt2_dst.scripts.preprocess_input \
    --input_path_json="${PATH_DATA_DIR}"/simmc_furniture/furniture_devtest_dials.json \
    --output_path_predict="${PATH_DIR}"/gpt2_dst/data/furniture/furniture_devtest_dials_predict.txt \
    --output_path_target="${PATH_DIR}"/gpt2_dst/data/furniture/furniture_devtest_dials_target.txt \
    --len_context=2 \
    --use_multimodal_contexts=1 \
    --input_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/furniture/special_tokens.json \

# Test split
python -m gpt2_dst.scripts.preprocess_input \
    --input_path_json="${PATH_DATA_DIR}"/simmc_furniture/furniture_devtest_dials_teststd_format_public.json \
    --output_path_predict="${PATH_DIR}"/gpt2_dst/data/furniture/furniture_test_dials_predict.txt \
    --output_path_target="${PATH_DIR}"/gpt2_dst/data/furniture/furniture_test_dials_target.txt \
    --len_context=2 \
    --use_multimodal_contexts=1 \
    --input_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/furniture/special_tokens.json \

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
python -m gpt2_dst.scripts.preprocess_input \
    --input_path_json="${PATH_DATA_DIR}"/simmc_fashion/fashion_devtest_dials_teststd_format_public.json \
    --output_path_predict="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_test_dials_predict.txt \
    --output_path_target="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_test_dials_target.txt \
    --len_context=2 \
    --use_multimodal_contexts=1 \
    --input_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/fashion/special_tokens.json \

echo "Done!"


# --------------------------------------------- #
#  STEP 2 : parse and lowercase belief states 	#
# ---------------------------------------------	#

echo 'Step 2 : Generating "parsed and lowercased belief state" data from original data...'

# Get special tokens, make a dictionary to swap special tokens to "parsed and lowercased" tokens

# Furniture
python -m gpt2_dst.scripts.make_token_dict \
	--domain=furniture

# Fashion
python -m gpt2_dst.scripts.make_token_dict \
	--domain=fashion

# Process original dataset into "parsed and lowercased" form

# back up original data
cp -r gpt2_dst/data/furniture gpt2_dst/data/furniture_original
cp -r gpt2_dst/data/fashion gpt2_dst/data/fashion_original

python -m gpt2_dst.scripts.make_parsed_data \
	--domain=furniture

python -m gpt2_dst.scripts.make_parsed_data \
	--domain=fashion

echo "Done!"

# --------------------------------------------- #
#  STEP 3 : dump train and dev data into total	#
# ---------------------------------------------	#
echo "Step 3 : dump train and dev data into total"

cat gpt2_dst/data/furniture/furniture_train_dials_predict.txt gpt2_dst/data/furniture/furniture_dev_dials_predict.txt \
	> gpt2_dst/data/furniture/furniture_total_dials_predict.txt

cat gpt2_dst/data/furniture/furniture_train_dials_target.txt gpt2_dst/data/furniture/furniture_dev_dials_target.txt \
	> gpt2_dst/data/furniture/furniture_total_dials_target.txt

cat gpt2_dst/data/fashion/fashion_train_dials_predict.txt gpt2_dst/data/fashion/fashion_dev_dials_predict.txt \
	> gpt2_dst/data/fashion/fashion_total_dials_predict.txt

cat gpt2_dst/data/fashion/fashion_train_dials_target.txt gpt2_dst/data/fashion/fashion_dev_dials_target.txt \
	> gpt2_dst/data/fashion/fashion_total_dials_target.txt

echo "Done!"
echo ""

echo "Preprocessing completed!"
