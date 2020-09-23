PATH_DIR=$(realpath .)
# PARAMETERS	DEFAUT VALUE
EPOCHS=1	# 1
TRAIN_BATCH=24	# 4
VAL_BATCH=32	# 4
MUL_GPU=0	# 1
GPU_ID='1'	# '0'
WARM_UP=2000	# 0
LR=5e-5		# 5e-5
LOGGING=3000	# 500:
PARAMETERS="EPOCHS		$EPOCHS
TRAIN_BATCH	$TRAIN_BATCH
VAL_BATCH	$VAL_BATCH
MUL_GPU		$MUL_GPU
N_GPU 		$N_GPU
"

python -m gpt2_dst.scripts.run_language_modeling \
    --output_dir="${PATH_DIR}"/gpt2_dst/save/fine_tune/ \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --line_by_line \
    --do_train \
    --train_data_file="${PATH_DIR}"/schema_data/output_total.txt \
    --do_eval \
    --eval_data_file="${PATH_DIR}"/schema_data/output_dev.txt \
    --evaluate_during_training \
    --logging_steps=$LOGGING \
    --num_train_epochs=$EPOCHS \
    --overwrite_output_dir \
    --learning_rate=$LR \
    --warmup_steps=$WARM_UP \
    --gpu_id=$GPU_ID \
    --mul_gpu=$MUL_GPU \
    --fp16 \
    --per_gpu_train_batch_size=$TRAIN_BATCH \
    --per_gpu_eval_batch_size=$VAL_BATCH