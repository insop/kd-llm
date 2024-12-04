#!/usr/bin/bash

set -e
set -u
set -o pipefail

EVAL_DIR="/workspace/outputs_kd-llm"
pushd $EVAL_DIR
MODEL=$1

echo -e "\033[0;32mRunning eval for Model: $MODEL\033[0m"

for eval_test in humaneval mbpp; do
    echo -e "\033[0;32mEVAL: $eval_test\033[0m"
    CMD="evalplus.evaluate $eval_test --model $MODEL  \
                --parallel $(nproc)  \
                --greedy --trust_remote_code=True \
                --force-base-prompt \
                --i-just-wanna-run"
    echo -e "\033[0;32m$CMD\033[0m"
    eval $CMD
    echo -e "\033[0;32mEval for Dataset ($eval_test): Done\033[0m"
done

popd