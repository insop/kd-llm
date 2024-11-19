set -ex

WORLD_SIZE=1
BATCH_SIZE=4
# BATCH_SIZE=4096
# MICRO_BATCH_SIZE=8
GRAD_ACCU=8

RUN_NAME="llama-3.2-1B_stack-smol_peft"
DATA_PATH="bigcode/the-stack-smol"
# DATA_PATH="bigcode/the-stack-dedup"
OUTPUT_PATH="$WORKSPACE/outputs/$RUN_NAME"
MODEL_PATH="meta-llama/Llama-3.2-1B"
LOG_PATH="$WORKSPACE/logs"

mkdir -p $OUTPUT_PATH
mkdir -p $LOG_PATH

pushd src
python ${DISTRIBUTED_ARGS} finetune.py \
    --bf16 \
    --data_path $DATA_PATH \
    --eval_strategy "no" \
    --gradient_accumulation_steps $GRAD_ACCU \
    --gradient_checkpointing \
    --learning_rate 2e-5 \
    --logging_dir $LOG_PATH \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --max_steps 10 \
    --model_max_length 2048 \
    --model_name_or_path $MODEL_PATH \
    --num_train_epochs 1 \
    --output_dir "$OUTPUT_PATH" \
    --per_device_eval_batch_size 1 \
    --report_to "wandb" \
    --run_name "$RUN_NAME" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 100 \
    --seq_length 2048 \
    --subset "data/python" \
    --warmup_steps 50
popd