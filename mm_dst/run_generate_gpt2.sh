#!/bin/bash

# parameters	defaut value
NUM_GEN=100000	# 100000
NUM_BEAMS=2	# 1
LENGTH=100	# 100
NGRAM=0		# 0
TOKEN=0		# 0
GPU_ID='1'	# '0'
TOP_P=1
TOP_K=1
TEMPERATURE=1
GPU_ID='0'	# '0'
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
	DOMAIN=$1
	KEYWORD=$2
fi

if [[ $# -eq 3 ]]
then
	DOMAIN=$1
	KEYWORD=$2
	MODEL=$3
fi

PATH_DIR=$(realpath .)
PATH_DATA_DIR=$(realpath ../data)

if [ $DOMAIN == "furniture_to" ] || [ $DOMAIN == "all" ]
then
# Generate sentences (Furniture, text-only)
python -m gpt2_dst.scripts.run_generation \
    --model_type=gpt2 \
    --model_name_or_path="${PATH_DIR}"/gpt2_dst/save/furniture_to/"${MODEL}" \
    --num_return_sequences=1 \
    --length=$LENGTH \
    --stop_token='<EOS>' \
    --num_beams=$NUM_BEAMS \
    --no_repeat_ngram_size=$NGRAM \
    --num_gen=$NUM_GEN \
    --token=$TOKEN \
    --gpu_id=$GPU_ID \
    --p=$TOP_P \
    --k=$TOP_K \
    --temperature=$TEMPERATURE \
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/furniture_to/furniture_devtest_dials_predict.txt \
    --path_output="${PATH_DIR}"/gpt2_dst/results/furniture_to/"${KEYWORD}"/furniture_devtest_dials_predicted.txt

echo "$PARAMETERS" > "${PATH_DIR}"/gpt2_dst/results/furniture_to/"${KEYWORD}"/parameters.txt
fi

if [ $DOMAIN == "furniture" ] || [ $DOMAIN == "all" ]
then
# Generate sentences (Furniture, multi-modal)
python -m gpt2_dst.scripts.run_generation \
    --model_type=gpt2 \
    --model_name_or_path="${PATH_DIR}"/gpt2_dst/save/furniture/"${MODEL}" \
    --num_return_sequences=1 \
    --length=$LENGTH \
    --stop_token='<EOS>' \
    --num_beams=$NUM_BEAMS \
    --no_repeat_ngram_size=$NGRAM \
    --num_gen=$NUM_GEN \
    --token=$TOKEN \
    --gpu_id=$GPU_ID \
    --p=$TOP_P \
    --k=$TOP_K \
    --temperature=$TEMPERATURE \
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/furniture/furniture_devtest_dials_predict.txt \
    --path_output="${PATH_DIR}"/gpt2_dst/results/furniture/"${KEYWORD}"/furniture_devtest_dials_predicted.txt
echo "$PARAMETERS" > "${PATH_DIR}"/gpt2_dst/results/furniture/"${KEYWORD}"/parameters.txt
fi

if [ $DOMAIN == "fashion_to" ] || [ $DOMAIN == "all" ]
then
# Generate sentences (Fashion, text-only)
python -m gpt2_dst.scripts.run_generation \
    --model_type=gpt2 \
    --model_name_or_path="${PATH_DIR}"/gpt2_dst/save/fashion_to/"${MODEL}" \
    --num_return_sequences=1 \
    --length=$LENGTH \
    --stop_token='<EOS>' \
    --num_beams=$NUM_BEAMS \
    --no_repeat_ngram_size=$NGRAM \
    --num_gen=$NUM_GEN \
    --token=$TOKEN \
    --gpu_id=$GPU_ID \
    --p=$TOP_P \
    --k=$TOP_K \
    --temperature=$TEMPERATURE \
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/fashion_to/fashion_devtest_dials_predict.txt \
    --path_output="${PATH_DIR}"/gpt2_dst/results/fashion_to/"${KEYWORD}"/fashion_devtest_dials_predicted.txt
echo "$PARAMETERS" > "${PATH_DIR}"/gpt2_dst/results/fashion_to/"${KEYWORD}"/parameters.txt
fi

if [ $DOMAIN == "fashion" ] || [ $DOMAIN == "all" ]
then
# Generate sentences (Fashion, multi-modal)
python -m gpt2_dst.scripts.run_generation \
    --model_type=gpt2 \
    --model_name_or_path="${PATH_DIR}"/gpt2_dst/save/fashion/"${MODEL}"  \
    --num_return_sequences=1 \
    --length=$LENGTH \
    --stop_token='<EOS>' \
    --num_beams=$NUM_BEAMS \
    --no_repeat_ngram_size=$NGRAM \
    --num_gen=$NUM_GEN \
    --token=$TOKEN \
    --gpu_id=$GPU_ID \
    --p=$TOP_P \
    --k=$TOP_K \
    --temperature=$TEMPERATURE \
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_devtest_dials_predict.txt \
    --path_output="${PATH_DIR}"/gpt2_dst/results/fashion/"${KEYWORD}"/fashion_devtest_dials_predicted.txt
echo "$PARAMETERS" > "${PATH_DIR}"/gpt2_dst/results/fashion/"${KEYWORD}"/parameters.txt
fi
