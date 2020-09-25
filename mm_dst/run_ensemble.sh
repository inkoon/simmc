PATH_DIR=$(realpath .)
DOMAIN="furniture"
python -m gpt2_dst.scripts.ensemble \
    --input_path_predicted_list "${PATH_DIR}"/predictions/furniture_devtest_dials_predicted_3.txt "${PATH_DIR}"/predictions/furniture_devtest_dials_predicted_4.txt "${PATH_DIR}"/predictions/furniture_devtest_dials_predicted_5.txt  \
    --output_path_ensembled="${PATH_DIR}"/gpt2_dst/results/ensembled.txt \
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/"${DOMAIN}"/"${DOMAIN}"_devtest_dials_predict.txt \
    --domain=furniture


echo "Evaluation of furniture_to/$KEYWORD : " 
# Evaluate (furniture, non-multimodal)
python -m gpt2_dst.scripts.evaluate \
    --input_path_target="${PATH_DIR}"/gpt2_dst/data/furniture/furniture_devtest_dials_target.txt \
    --input_path_predicted="${PATH_DIR}"/gpt2_dst/results/ensembled.txt \
    --output_path_report="${PATH_DIR}"/gpt2_dst/results/ensembled_devtest_dials_report.json