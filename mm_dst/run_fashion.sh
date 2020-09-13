#!/bin/bash
if [[ $# -lt 1 ]]
then
    KEY_WORD=""
else
    KEY_WORD="_$1"
fi

PATH_DIR=$(realpath .)
PATH_DATA_DIR=$(realpath ../data)

# PARAMETERS
CONTEXT=3
GPU_ID='1'
RESPONCE=1
STATE=1
BEAMS=1



echo "Running on GPU $GPU_ID with keyword $KEY_WORD"

# fashion
# Multimodal Data
# Train split
python -m gpt2_dst.scripts.preprocess_input \
    --input_path_json="${PATH_DATA_DIR}"/simmc_fashion/fashion_train_dials.json \
    --output_path_predict="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_train_dials_predict$KEY_WORD.txt \
    --output_path_target="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_train_dials_target$KEY_WORD.txt \
    --len_context=$CONTEXT \
    --use_multimodal_contexts=1 \
    --output_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/fashion/special_tokens$KEY_WORD.json \
    --gpu_id=$GPU_ID

# Dev split
python -m gpt2_dst.scripts.preprocess_input \
    --input_path_json="${PATH_DATA_DIR}"/simmc_fashion/fashion_dev_dials.json \
    --output_path_predict="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_dev_dials_predict$KEY_WORD.txt \
    --output_path_target="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_dev_dials_target$KEY_WORD.txt \
    --len_context=$CONTEXT \
    --use_multimodal_contexts=1 \
    --input_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/fashion/special_tokens$KEY_WORD.json \
    --output_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/fashion/special_tokens$KEY_WORD.json \
    --gpu_id=$GPU_ID

# Devtest split
python -m gpt2_dst.scripts.preprocess_input \
    --input_path_json="${PATH_DATA_DIR}"/simmc_fashion/fashion_devtest_dials.json \
    --output_path_predict="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_devtest_dials_predict$KEY_WORD.txt \
    --output_path_target="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_devtest_dials_target$KEY_WORD.txt \
    --len_context=$CONTEXT \
    --use_multimodal_contexts=1 \
    --input_path_special_tokens="${PATH_DIR}"/gpt2_dst/data/fashion/special_tokens$KEY_WORD.json \
    --gpu_id=$GPU_ID

# Train (fashion, multi-modal)
python -m gpt2_dst.scripts.run_language_modeling \
    --output_dir="${PATH_DIR}"/gpt2_dst/save/fashion$KEY_WORD \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --line_by_line \
    --add_special_tokens="${PATH_DIR}"/gpt2_dst/data/fashion/special_tokens$KEY_WORD.json \
    --do_train \
    --train_data_file="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_train_dials_target$KEY_WORD.txt \
    --do_eval \
    --eval_data_file="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_dev_dials_target$KEY_WORD.txt \
    --num_train_epochs=8 \
    --overwrite_output_dir \
    --per_gpu_train_batch_size=8 \
    --per_gpu_eval_batch_size=8 \
    --warmup_steps=1000 \
    --save_steps=1000 \
    --gpu_id=$GPU_ID 

# Generate sentences (fashion, multi-modal)
python -m gpt2_dst.scripts.run_generation \
    --model_type=gpt2 \
    --model_name_or_path="${PATH_DIR}"/gpt2_dst/save/fashion$KEY_WORD/ \
    --num_return_sequences=1 \
    --length=100 \
    --stop_token='<EOS>' \
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_devtest_dials_predict.txt \
    --path_output="${PATH_DIR}"/gpt2_dst/results/fashion/fashion_devtest_dials_predicted$KEY_WORD.txt \
    --num_beams=$BEAMS \
    --gpu_id=$GPU_ID

# Evaluate (fashion, multi-modal)
python -m gpt2_dst.scripts.evaluate \
    --input_path_target="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_devtest_dials_target.txt \
    --input_path_predicted="${PATH_DIR}"/gpt2_dst/results/fashion/fashion_devtest_dials_predicted$KEY_WORD.txt \
    --output_path_report="${PATH_DIR}"/gpt2_dst/results/fashion/fashion_devtest_dials_report$KEY_WORD.json