#!/bin/bash
if [[ $# -eq 0 ]]
then
	echo "ERROR : run format > $./*.sh (DOMAIN) KEYWORD"
	exit 1
fi

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

PATH_DIR=$(realpath .)

if [ $DOMAIN == "furniture_to" ] || [ $DOMAIN == "all" ]
then
echo ""
echo "Evaluation of furniture_to/$KEYWORD : " 
# Evaluate (furniture, non-multimodal)
python -m gpt2_dst.scripts.evaluate \
    --input_path_target="${PATH_DIR}"/gpt2_dst/data/furniture_to/furniture_devtest_dials_target.txt \
    --input_path_predicted="${PATH_DIR}"/gpt2_dst/results/furniture_to/"${KEYWORD}"/furniture_devtest_dials_predicted.txt \
    --output_path_report="${PATH_DIR}"/gpt2_dst/results/furniture_to/"${KEYWORD}"/furniture_devtest_dials_report.json
fi

if [ $DOMAIN == "furniture" ] || [ $DOMAIN == "all" ]
then
echo ""
echo "Evaluation of furniture/$KEYWORD : " 
# Evaluate (furniture, multi-modal)
python -m gpt2_dst.scripts.evaluate \
    --input_path_target="${PATH_DIR}"/gpt2_dst/data/furniture/furniture_devtest_dials_target.txt \
    --input_path_predicted="${PATH_DIR}"/gpt2_dst/results/furniture/"${KEYWORD}"/furniture_devtest_dials_predicted.txt \
    --output_path_report="${PATH_DIR}"/gpt2_dst/results/furniture/"${KEYWORD}"/furniture_devtest_dials_report.json
fi

if [ $DOMAIN == "fashion_to" ] || [ $DOMAIN == "all" ]
then
echo ""
echo "Evaluation of fashion_to/$KEYWORD : " 
# Evaluate (Fashion, non-multimodal)
python -m gpt2_dst.scripts.evaluate \
    --input_path_target="${PATH_DIR}"/gpt2_dst/data/fashion_to/fashion_devtest_dials_target.txt \
    --input_path_predicted="${PATH_DIR}"/gpt2_dst/results/fashion_to/"${KEYWORD}"/fashion_devtest_dials_predicted.txt \
    --output_path_report="${PATH_DIR}"/gpt2_dst/results/fashion_to/"${KEYWORD}"/fashion_devtest_dials_report.json
fi

if [ $DOMAIN == "fashion" ] || [ $DOMAIN == "all" ]
then
echo ""
echo "Evaluation of fashion/$KEYWORD : " 
# Evaluate (Fashion, multi-modal)
python -m gpt2_dst.scripts.evaluate \
    --input_path_target="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_devtest_dials_target.txt \
    --input_path_predicted="${PATH_DIR}"/gpt2_dst/results/fashion/"${KEYWORD}"/fashion_devtest_dials_predicted.txt \
    --output_path_report="${PATH_DIR}"/gpt2_dst/results/fashion/"${KEYWORD}"/fashion_devtest_dials_report.json
fi
