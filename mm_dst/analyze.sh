
PATH_DIR=$(realpath .)
DOMAIN='furniture'
python -m gpt2_dst.scripts.analysis \
    --input_path_target="${PATH_DIR}"/gpt2_dst/data/${DOMAIN}/furniture_devtest_dials_target.txt \
    --input_path_predicted="${PATH_DIR}"/gpt2_dst/results/${DOMAIN}/"${KEYWORD}"/${DOMAIN}_devtest_dials_predicted.txt \
    --output_path_report="${PATH_DIR}"/gpt2_dst/results/${DOMAIN}/"${KEYWORD}"/"${DOMAIN}"_devtest_dials_analyze.json \
    --limit=0.1