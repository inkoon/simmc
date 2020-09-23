#!/bin/bash
PATH_DIR=$(realpath .)
if [[ $# -eq 0 ]] 
then
	echo "run format > ./convert_all.sh [input_folder]  (CONTEXT)"
	exit 1
fi


if [[ $# -eq 1 ]]
then
    INPUT_FOLDER=$1
    CONTEXT=2
    NAME="output.txt"
fi

if [[ $# -eq 3 ]]
then
    INPUT_FOLDER=$1
    CONTEXT=$2
    NAME=$3
fi
echo "Start to extract files from $INPUT_FOLDER"
for FILE in $INPUT_FOLDER/*;
    do
        if [ $FILE == "$INOUT_FOLDER/schema.json" ]  
        then 
            break
        fi 
        python -m convert \
--input_path_json=$FILE \
--output_path=$NAME \
--context=$CONTEXT 
    echo "..done converting data from $FILE to SIMMC data format"
done 
