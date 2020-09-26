#!/bin/bash
# parameters	defaut value
NUM_GEN=500 # 100000
NUM_BEAMS=2	# 1
LENGTH=500	# 100
NGRAM=0		# 0
TOKEN=0		# 0
GPU_ID='2'	# '0'
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
    PATH_DIR=$(realpath .)
fi

# Generate sentences (Furniture, multi-modal)
python -m gpt2_dst.scripts.run_generation_en \
    --model_type=gpt2 \
    --model_name_or_path_list "${PATH_DIR}"/gpt2_dst/save/furniture/large "${PATH_DIR}"/gpt2_dst/save/furniture/td_small "${PATH_DIR}"/gpt2_dst/save/furniture/small \
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
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/furniture/furniture_devtest_dials_predict.txt \
    --path_output="${PATH_DIR}"/gpt2_dst/results/furniture/toy_ensembled_all_furniture_train_dials_predicted.txt
#echo "$PARAMETERS" > "${PATH_DIR}"/gpt2_dst/results/furniture/"${KEYWORD}"/parameters.txt
