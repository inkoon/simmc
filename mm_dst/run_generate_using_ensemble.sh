#!/bin/bash
# parameters	defaut value
NUM_GEN=100000 # 100000
NUM_BEAMS=2	# 1
LENGTH=100	# 100
NGRAM=0		# 0
TOKEN=0		# 0
GPU_ID='2'	# '0'
TOP_P=0.9	# 0.9
TOP_K=0		# 0
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


if [[ $# -eq 1 ]]
then
	DOMAIN=$1
fi

PATH_DIR="/home/boychaboy/simmc/mm_dst" 

if [ $DOMAIN == "furniture" ]
then 
GPU_ID='0'
# Generate sentences (Furniture, multi-modal)
python -m gpt2_dst.scripts.run_generation_en \
    --model_type=gpt2 \
    --model_name_or_path_list "${PATH_DIR}"/gpt2_dst/save/furniture/large/large "${PATH_DIR}"/gpt2_dst/save/furniture/td_large \
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
    --path_output="${PATH_DIR}"/gpt2_dst/results/furniture/ensembled_furniture_devtest_dials_predicted.txt
fi


if [ $DOMAIN == "furniture_to" ]
then 
GPU_ID='0'
# Generate sentences (Furniture, text only)
python -m gpt2_dst.scripts.run_generation_en \
    --model_type=gpt2 \
    --model_name_or_path_list "${PATH_DIR}"/gpt2_dst/save/furniture_to/large "${PATH_DIR}"/gpt2_dst/save/furniture_to/td_large \
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
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/furniture_to/furniture_devtest_dials_predict.txt \
    --path_output="${PATH_DIR}"/gpt2_dst/results/furniture_to/ensembled_furniture_devtest_dials_predicted.txt
fi




if [ $DOMAIN == "fashion" ]
then 
GPU_ID='1'
# Generate sentences (Fashion, multi-modal)
python -m gpt2_dst.scripts.run_generation_en \
    --model_type=gpt2 \
    --model_name_or_path_list "${PATH_DIR}"/gpt2_dst/save/fashion/td_small "${PATH_DIR}"/gpt2_dst/save/fashion/finetune "${PATH_DIR}"/gpt2_dst/save/fashion/td_large \
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
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_devtest_dials_predict.txt \
    --path_output="${PATH_DIR}"/gpt2_dst/results/fashion/ensembled_fashion_devtest_dials_predicted.txt
fi




if [ $DOMAIN == "fashion_to" ]
then 
GPU_ID='1'
# Generate sentences (Fashion, multi-modal)
python -m gpt2_dst.scripts.run_generation_en \
    --model_type=gpt2 \
    --model_name_or_path_list "${PATH_DIR}"/gpt2_dst/save/fashion_to/td_large "${PATH_DIR}"/gpt2_dst/save/fashion_to/medium 
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
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/fashion_to/fashion_devtest_dials_predict.txt \
    --path_output="${PATH_DIR}"/gpt2_dst/results/fashion_to/ensembled_fashion_devtest_dials_predicted.txt
fi



#echo "$PARAMETERS" > "${PATH_DIR}"/gpt2_dst/results/furniture/"${KEYWORD}"/parameters.txt

