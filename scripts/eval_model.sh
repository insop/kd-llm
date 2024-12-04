#!/usr/bin/bash
# 

# leetcode Pass@1: 0.016666666666666666
# bigcodebench Pass@1: 0.03245614035087719
# export MODEL_DIR="/workspace/outputs_kd-llm/kd_11_25-code_alpaca_120k-ft-Meta-Llama-3.2-1B-ft-teacher_3.1-8B-1"

# export MODEL_DIR="/workspace/outputs_kd-llm/kd_11_25-stack-Meta-Llama-3.2-1B-2"

# export MODEL_DIR="/workspace/outputs_kd-llm/kd_11_25-stack-1/Meta-Llama-3.2-1B"

# kd 1b lora tuning trained with alpaca-code-120k, with FT teacher 8b
# bigcodebench  Pass@1: 0.037719298245614034
# Pass@1: 0.21341463414634146
export MODEL_DIR="/workspace/outputs_kd-llm/kd_11_25-code_alpaca_120k-Meta-Llama-3.2-1B-ft-teacher_3.1-8B-1"


# kd 1b lora tuning trained with alpaca-code-120k, bad...
# plus
# Pass@1: 0.18292682926829268
# export MODEL_DIR="/workspace/outputs_kd-llm/kd_11_25-code_alpaca_120k-Meta-Llama-3.2-1B-teacher_3.1-8B-1"

#1b ft lora tuning trained with alpaca-code-120k
# bigcodebench Pass@1: 0.037719298245614034
# plus, Pass@1: 0.17682926829268292
# Pass@1: 0.22560975609756098
# export MODEL_DIR="/workspace/outputs_kd-llm/1b_lora-Llama-3.2-1B-alpaca-code-120k"

#8b ft lora tuning trained with alpaca-code-120k
# plus, Pass@1: 0.34146341463414637
# Pass@1: 0.4024390243902439
# export MODEL_DIR="/workspace/outputs_kd-llm/8b_lora-Llama-3.1-8B-alpaca-code-120k"

#8b ft lora tuning trained with alpaca-code-20k
# Pass@1: 0.38414634146341464
# export MODEL_DIR="/workspace/outputs_kd-llm/8b_lora-Llama-3.1-8B-alpaca-code"

#8b ft lora tuning trained with alpaca-clean!! no code
# Pass@1: 0.39634146341463417
# export MODEL_DIR="/workspace/outputs_kd-llm/8b_lora-Llama-3.1-8B-alpaca-clean"

# stack-smol ????
# Pass@1: 0.20121951219512196
# export MODEL_DIR="/workspace/outputs_kd-llm/kd_11_25-stack-smol-Meta-Llama-3.2-1B-teacher_3.1-8B-test"

# Pass@1: 0.18902439024390244
# export MODEL_DIR="/workspace/outputs_kd-llm/kd_11_25-stack-Meta-Llama-3.2-1B-teacher_3.1-8B"

# Pass@1: 0.3902439024390244
# export MODEL_DIR="meta-llama/Llama-3.1-8B"

# Pass@1: 0.2865853658536585
# export MODEL_DIR="meta-llama/Llama-3.2-3B"

# leetcode: Pass@1: 0.016666666666666666
# bigcodebench: Pass@1: 0.034210526315789476
# plus, Pass@1: 0.17682926829268292
# Pass@1: 0.20121951219512196
# export MODEL_DIR="meta-llama/Llama-3.2-1B"

export TEST_NAME=$(basename "${MODEL_DIR}")
export MODEL="${MODEL_DIR}"

echo "Model: $MODEL"
echo "Test_name: $TEST_NAME"

pushd /workspace/OpenCoder-llm/OpenCodeEval

python src/main.py \
    --model_name $MODEL \
    --task "BigCodeBench" \
    --save "test/{$TEST_NAME}" \
    --backend "vllm" \
    --num_gpus 4 \
    --num_samples 1 \
    --temperature 0.0 \
    --num_workers 10 \
    --batch_size 200 \
    --max_tokens 4096 \
    --model_type "Base" \
    --prompt_type "Completion" \
    --prompt_prefix "" \
    --prompt_suffix "" \
    --trust_remote_code
popd