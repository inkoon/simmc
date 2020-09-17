
PATH_DIR=$(realpath .)
DOMAIN='furniture'
#DOMAIN='fashion'
python -m gpt2_dst.scripts.analysis \
    --input_path_target="${PATH_DIR}"/gpt2_dst/data/${DOMAIN}/${DOMAIN}_devtest_dials_target.txt \
    --input_path_predicted="${PATH_DIR}"/gpt2_dst/results/${DOMAIN}/${DOMAIN}_devtest_dials_predicted.txt \
    --output_dir="${PATH_DIR}"/gpt2_dst/results/${DOMAIN}/analysis \
    --limit=0.1
