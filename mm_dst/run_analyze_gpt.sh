PATH_DIR=$(realpath .)

if [[ $# -eq 0 ]] 
then
	echo "run format > ./run_analyze.sh (DOMAIN)  [keyword]"
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

_='
T_FURN="furniture"
T_FURN_T="furniture_to"
T_FA="fashion"
T_FA_T="fashion_to"
'
T_FURN="toy_furniture"
T_FURN_T="toy_furniture_to"
T_FA="toy_fashion"
T_FA_T="toy_fashion_to"

if [ $DOMAIN == "furniture_to" ] || [ $DOMAIN == "all" ]
then
echo "Quick summary for furniture to for keyword $KEYWORD"
python -m gpt2_dst.scripts.analysis \
    --input_path_target="${PATH_DIR}"/gpt2_dst/data/$T_FURN_T/furniture_devtest_dials_target.txt \
    --input_path_predicted="${PATH_DIR}"/gpt2_dst/results/furniture_to/$KEYWORD/furniture_to_devtest_dials_predicted.txt \
    --output_dir="${PATH_DIR}"/gpt2_dst/results/furniture_to/$KEYWORD/analysis \
    --limit=0.1
fi

if [ $DOMAIN == "furniture" ] || [ $DOMAIN == "all" ]
then
    echo "\n\nQuick summary furniture for keyword $KEYWORD"
python -m gpt2_dst.scripts.analysis \
    --input_path_target="${PATH_DIR}"/gpt2_dst/data/$T_FURN/furniture_devtest_dials_target.txt \
    --input_path_predicted="${PATH_DIR}"/gpt2_dst/results/furniture/$KEYWORD/furniture_devtest_dials_predicted.txt \
    --output_dir="${PATH_DIR}"/gpt2_dst/results/furniture/$KEYWORD/analysis \
    --limit=0.1
fi


if [ $DOMAIN == "fashion_to" ] || [ $DOMAIN == "all" ]
then
    echo "\n\nQuick summary fashion_to for keyword $KEYWORD"
    python -m gpt2_dst.scripts.analysis \
    --input_path_target="${PATH_DIR}"/gpt2_dst/data/$T_FA_T/fashion_devtest_dials_target.txt \
    --input_path_predicted="${PATH_DIR}"/gpt2_dst/results/fashion_to/$KEYWORD/fashion_devtest_dials_predicted.txt \
    --output_dir="${PATH_DIR}"/gpt2_dst/results/fashion_to/$KEYWORD/analysis \
    --limit=0.1
fi


if [ $DOMAIN == "fashion" ] || [ $DOMAIN == "all" ]
then
    echo "\n\nQuick summary for fashion for keyword $KEYWORD"
  python -m gpt2_dst.scripts.analysis \
    --input_path_target="${PATH_DIR}"/gpt2_dst/data/$T_FA/fashion_devtest_dials_target.txt \
    --input_path_predicted="${PATH_DIR}"/gpt2_dst/results/fashion/$KEYWORD/fashion_devtest_dials_predicted.txt \
    --output_dir="${PATH_DIR}"/gpt2_dst/results/fashion/$KEYWORD/analysis \
    --limit=0.1
fi
