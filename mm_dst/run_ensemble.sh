#!/bin/bash
if [[ $# -eq 0 ]]
then
	DOMAIN="all"
fi

if [[ $# -eq 1 ]]
then
	DOMAIN=$1
fi

PATH_DIR=$(realpath .)
if [ $DOMAIN == "furniture" ] || [ $DOMAIN == "all" ]
then
python -m gpt2_dst.scripts.ensemble \
    --input_path_predicted_list  "${PATH_DIR}"/predictions/furniture/large.txt "${PATH_DIR}"/predictions/furniture/small.txt "${PATH_DIR}"/predictions/furniture/td_large_unparsed.txt --output_path_ensembled="${PATH_DIR}"/gpt2_dst/results/furniture/ensembled.txt \
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/"${DOMAIN}"/"${DOMAIN}"_devtest_dials_predict.txt \
    --domain=furniture
# ~/predictions/td_small.txt \
# "${PATH_DIR}"/predictions/furniture_devtest_dials_predicted_3.txt "${PATH_DIR}"/predictions/furniture_devtest_dials_predicted_4.txt "${PATH_DIR}"/predictions/furniture_devtest_dials_predicted_5.txt
echo "Evaluation of furniture : " 
# Evaluate (furniture, non-multimodal)
python -m gpt2_dst.scripts.evaluate \
    --input_path_target="${PATH_DIR}"/gpt2_dst/data/furniture/furniture_devtest_dials_target.txt \
    --input_path_predicted="${PATH_DIR}"/gpt2_dst/results/furniture/ensembled.txt \
    --output_path_report="${PATH_DIR}"/gpt2_dst/results/furniture/ensembled_devtest_dials_report.json
fi

if [ $DOMAIN == "furniture_to" ] || [ $DOMAIN == "all" ]
then
python -m gpt2_dst.scripts.ensemble \
    --input_path_predicted_list  "${PATH_DIR}"/predictions/furniture_to/td_large.txt "${PATH_DIR}"/predictions/furniture_to/td_small.txt --output_path_ensembled="${PATH_DIR}"/gpt2_dst/results/furniture_to/furniture_ensembled.txt \
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/furniture_to/furniture_devtest_dials_predict.txt \
    --domain=furniture                   # td_finetune.txt  td_large.txt  td_small.txt
# ~/predictions/td_small.txt \
# "${PATH_DIR}"/predictions/furniture_devtest_dials_predicted_3.txt "${PATH_DIR}"/predictions/furniture_devtest_dials_predicted_4.txt "${PATH_DIR}"/predictions/furniture_devtest_dials_predicted_5.txt
echo "Evaluation of furniture_to: " 
# Evaluate (furniture, non-multimodal)
python -m gpt2_dst.scripts.evaluate \
    --input_path_target="${PATH_DIR}"/gpt2_dst/data/furniture_to/furniture_devtest_dials_target.txt \
    --input_path_predicted="${PATH_DIR}"/gpt2_dst/results/furniture_to/furniture_ensembled.txt \
    --output_path_report="${PATH_DIR}"/gpt2_dst/results/furniture_to/ensembled_devtest_dials_report.json
fi

if [ $DOMAIN == "fashion" ] || [ $DOMAIN == "all" ]
then
python -m gpt2_dst.scripts.ensemble \
    --input_path_predicted_list  "${PATH_DIR}"/predictions/fashion/large.txt "${PATH_DIR}"/predictions/fashion/td_small.txt "${PATH_DIR}"/predictions/fashion/small.txt --output_path_ensembled="${PATH_DIR}"/gpt2_dst/results/fashion/fashion_ensembled.txt \
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_devtest_dials_predict.txt \
    --domain=fashion
# ~/predictions/td_small.txt \
# "${PATH_DIR}"/predictions/furniture_devtest_dials_predicted_3.txt "${PATH_DIR}"/predictions/furniture_devtest_dials_predicted_4.txt "${PATH_DIR}"/predictions/furniture_devtest_dials_predicted_5.txt
echo "Evaluation of fashion : " 
# Evaluate (furniture, non-multimodal)
python -m gpt2_dst.scripts.evaluate \
    --input_path_target="${PATH_DIR}"/gpt2_dst/data/fashion/fashion_devtest_dials_target.txt \
    --input_path_predicted="${PATH_DIR}"/gpt2_dst/results/fashion/fashion_ensembled.txt \
    --output_path_report="${PATH_DIR}"/gpt2_dst/results/fashion/fashion_ensembled_devtest_dials_report.json
fi

if [ $DOMAIN == "fashion_to" ] || [ $DOMAIN == "all" ]
then
python -m gpt2_dst.scripts.ensemble \
    --input_path_predicted_list  "${PATH_DIR}"/predictions/fashion_to/td_large.txt "${PATH_DIR}"/predictions/fashion_to/medium.txt "${PATH_DIR}"/predictions/fashion_to/small.txt --output_path_ensembled="${PATH_DIR}"/gpt2_dst/results/fashion_to/fashion_to_ensembled.txt \
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/fashion_to/fashion_devtest_dials_predict.txt \
    --domain=fashion                                # medium.txt  small.txt  td_large.txt
# ~/predictions/td_small.txt \
# "${PATH_DIR}"/predictions/furniture_devtest_dials_predicted_3.txt "${PATH_DIR}"/predictions/furniture_devtest_dials_predicted_4.txt "${PATH_DIR}"/predictions/furniture_devtest_dials_predicted_5.txt
echo "Evaluation of fashion_to : " 
# Evaluate (furniture, non-multimodal)
python -m gpt2_dst.scripts.evaluate \
    --input_path_target="${PATH_DIR}"/gpt2_dst/data/fashion_to/fashion_devtest_dials_target.txt \
    --input_path_predicted="${PATH_DIR}"/gpt2_dst/results/fashion_to/fashion_to_ensembled.txt \
    --output_path_report="${PATH_DIR}"/gpt2_dst/results/fashion_to/ensembled_devtest_dials_report.json
fi