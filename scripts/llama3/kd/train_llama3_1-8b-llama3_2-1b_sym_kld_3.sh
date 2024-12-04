#!/usr/bin/bash

set -e
set -u
set -o pipefail


DESC="Symmetric KLD"
echo -e "\033[0;32m$DESC\033[0m"

# Env setup
WORK_DIR=/workspace/workspace
CONFIG_DIR="$WORK_DIR/configs/llama3_2"
OUTPUT_BASE_DIR="/workspace/outputs_kd-llm/"
TOKENIZER_DIR="$OUTPUT_BASE_DIR/tokens_llama-3.2/"
TIMESTAMP=$(date +%Y%m%d-%H%M)

# =========
# TORCHTUNE
# update
# =========
TRAIN_METHOD=knowledge_distillation_distributed
CONFIG_FILE="kd/kd_llama3_2_sym_kld_config.yaml"

# =========
## EXPERIMENT PARAMETERS
# update
# =========
# KD CONFIG
KD_RATIO=0.75
SYM_KD_RATIO=0.5

# =========
## MODEL
# update
# =========
# STUDENT MODEL
CHECKPOINT_DIR="/workspace/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/"
# TEACHER MODEL
TEACHER_CHECKPOINT_DIR="/workspace/outputs_kd-llm/8b_lora-Llama-3.1-8B-alpaca-code-120k"

# =========
# take the file name and use as a train name
TRAIN_NAME="${BASH_SOURCE[0]##*/}"
TRAIN_NAME="${TRAIN_NAME%.*}"


# dist
NNMODES=1
NPROC_PER_NODE=7
DIST_TRAIN_OPTIONS="--nnodes $NNMODES --nproc_per_node $NPROC_PER_NODE"

# =========

MODEL_NAME=$(basename "$CHECKPOINT_DIR")
CHECKPOINT_OUTPUT_DIR="$OUTPUT_BASE_DIR/${TRAIN_NAME}-$TIMESTAMP"

TEACHER_MODEL=$(basename "$TEACHER_CHECKPOINT_DIR")

CONFIG_FILE_OPTION="--config $CONFIG_DIR/$CONFIG_FILE"

CONFIG_OPTIONS=""
CONFIG_OPTIONS+=" checkpointer.checkpoint_dir=$CHECKPOINT_DIR"
CONFIG_OPTIONS+=" checkpointer.output_dir=$CHECKPOINT_OUTPUT_DIR"
CONFIG_OPTIONS+=" teacher_checkpointer.checkpoint_dir=$TEACHER_CHECKPOINT_DIR"
CONFIG_OPTIONS+=" kd_ratio=$KD_RATIO"
CONFIG_OPTIONS+=" sym_kd_ratio=$SYM_KD_RATIO"

TUNE_CMD="tune run"

mkdir -p "$CHECKPOINT_OUTPUT_DIR"
touch "$CHECKPOINT_OUTPUT_DIR/.inprogress"

CMD="$TUNE_CMD $DIST_TRAIN_OPTIONS $TRAIN_METHOD $CONFIG_FILE_OPTION $CONFIG_OPTIONS"

echo -e "\033[32mModel: $MODEL_NAME\033[0m"
echo -e "\033[33mTeacher model: $TEACHER_MODEL\033[0m"
echo -e "\033[34mConfig options: $CONFIG_OPTIONS\033[0m"
echo -e "\033[35mCheckpoint output dir: $CHECKPOINT_OUTPUT_DIR\033[0m"
echo -e "\033[36mDist kd_ratio: $KD_RATIO\033[0m"

echo -e "\033[36mCOMMNEND: $CMD\033[0m"

$CMD

rm -f "$CHECKPOINT_OUTPUT_DIR/.inprogress"
touch "$CHECKPOINT_OUTPUT_DIR/.done"

echo -e "\033[31mTRAINING DONE\033[0m"

# prepare for tokenizer

mkdir -p "$CHECKPOINT_OUTPUT_DIR"
cp "$TOKENIZER_DIR"* "$CHECKPOINT_OUTPUT_DIR"


# running eval

EVAL_CMD="$WORK_DIR/scripts/llama3/eval/run_eval.sh $CHECKPOINT_OUTPUT_DIR"

echo -e "\033[32mEVAL START\033[0m"
echo -e "\033[33m========\033[0m"
echo $EVAL_CMD
$EVAL_CMD
echo -e "\033[34mEVAL DONE\033[0m"
