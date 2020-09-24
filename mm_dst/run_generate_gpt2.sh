#!/bin/bash
# parameters	defaut value
NUM_GEN=500 # 100000
NUM_BEAMS=2	# 1
LENGTH=100	# 100
NGRAM=0		# 0
TOKEN=0		# 0
GPU_ID='1'	# '0'
TOP_P=1.0 	# 0.9
TOP_K=5		# 0
TEMPERATURE=1	# 1

PARAMETERS="NUM_GEN		$NUM_GEN
NUM_BEAMS	$NUM_BEAMS
LENGTH		$LENGTH
NGRAM		$NGRAM
TOKEN 		$TOKEN
TOP_P		$TOP_P
TOP_K		$TOP_K
TEMPERATURE   	$TEMPERATURE"

if [[ $# -eq 0 ]]
then
	echo "error : run format > ./*.sh (domain) keyword"
	exit 1
fi

MODEL=""

if [[ $# -eq 1 ]]
then
	DOMAIN="all"
	KEYWORD=$1
fi

if [[ $# -eq 2 ]]
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
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/furniture_to/furniture_devtest_dials_predict.txt \
    --path_output="${PATH_DIR}"/gpt2_dst/results/furniture_to/furniture_devtest_dials_predicted.txt

# Generate sentences (Furniture, multi-modal)
python -m gpt2_dst.scripts.run_generation \
    --model_type=gpt2 \
    --model_name_or_path="${PATH_DIR}"/gpt2_dst/save/furniture/ \
    --num_return_sequences=1 \
    --length=100 \
    --stop_token='<EOS>' \
    --num_beams=$NUM_BEAMS \
    --no_repeat_ngram_size=$NGRAM \
    --num_gen=$NUM_GEN \
    --token=$TOKEN \
    --gpu_id=$GPU_ID \
    --p=$TOP_P \
    --k=$TOP_K \
    --gpu_id=$GPU_ID \
    --temperature=$TEMPERATURE \
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/furniture/furniture_train_dials_predict.txt \
    --path_output="${PATH_DIR}"/gpt2_dst/results/furniture/"${KEYWORD}"/furniture_train_dials_predicted.txt
#echo "$PARAMETERS" > "${PATH_DIR}"/gpt2_dst/results/furniture/"${KEYWORD}"/parameters.txt

# Generate sentences (Fashion, text-only)
python -m gpt2_dst.scripts.run_generation \
    --model_type=gpt2 \
    --model_name_or_path="${PATH_DIR}"/gpt2_dst/save/fashion_to/ \
    --num_return_sequences=1 \
    --length=100 \
    --stop_token='<EOS>' \
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/fashion_to/fashion_devtest_dials_predict.txt \
    --path_output="${PATH_DIR}"/gpt2_dst/results/fashion_to/fashion_devtest_dials_predicted.txt

# Generate sentences (Fashion, multi-modal)
python -m gpt2_dst.scripts.run_generation \
    --model_type=gpt2 \
    --model_name_or_path="${PATH_DIR}"/gpt2_dst/save/fashion/ \
    --num_return_sequences=1 \
    --length=100 \
    --stop_token='<EOS>' \
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_devtest_dials_predict.txt \
    --path_output="${PATH_DIR}"/gpt2_dst/results/fashion/fashion_devtest_dials_predicted.txt
