#!/usr/bin/bash

# mbpp (base tests)
# pass@1: 0.368
# mbpp+ (base + extra tests)
# pass@1: 0.317
# humaneval (base tests)
# pass@1: 0.201
# humaneval+ (base + extra tests)
# pass@1: 0.165
# export MODEL="/workspace/outputs_kd-llm/kd_11_25-evol-codealpaca_v1-ft-Meta-Llama-3.2-1B-ft-teacher_3.1-8B-1/"

# mbpp (base tests)
# pass@1: 0.362
# mbpp+ (base + extra tests)
# pass@1: 0.294
# humaneval (base tests)
# pass@1: 0.226
# humaneval+ (base + extra tests)
# pass@1: 0.183
# export MODEL="/workspace/outputs_kd-llm/kd_11_25-code_alpaca_120k-ft-Meta-Llama-3.2-1B-ft-teacher_3.1-8B-1"

# mbpp (base tests)
# pass@1: 0.410
# mbpp+ (base + extra tests)
# pass@1: 0.336
# humaneval (base tests)
# pass@1: 0.207
# humaneval+ (base + extra tests)
# pass@1: 0.177
# export MODEL="/workspace/outputs_kd-llm/kd_11_25-code_alpaca_120k-Meta-Llama-3.2-1B-ft-teacher_3.1-8B-1"

# mbpp (base tests)
# pass@1: 0.381
# mbpp+ (base + extra tests)
# pass@1: 0.325
# humaneval (base tests) 
# pass@1: 0.189
# humaneval+ (base + extra tests)
# pass@1: 0.146 
# export MODEL="/workspace/outputs_kd-llm/kd_11_25-code_alpaca_120k-Meta-Llama-3.2-1B-teacher_3.1-8B-1"

# mbpp (base tests)
# pass@1: 0.643
# mbpp+ (base + extra tests)
# pass@1: 0.532
# humaneval (base tests)
# pass@1: 0.390
# humaneval+ (base + extra tests)
# pass@1: 0.341
# export MODEL="/workspace/outputs_kd-llm/8b_lora-Llama-3.1-8B-alpaca-code-120k"

# mbpp (base tests)
# pass@1: 0.376
# mbpp+ (base + extra tests)
# pass@1: 0.317
# humaneval (base tests)
# pass@1: 0.220
# humaneval+ (base + extra tests)
# pass@1: 0.177
# export MODEL="/workspace/outputs_kd-llm/1b_lora-Llama-3.2-1B-alpaca-code-120k"

# mbpp (base tests)
# pass@1: 0.616
# mbpp+ (base + extra tests)
# pass@1: 0.516
# humaneval (base tests)
# pass@1: 0.384
# humaneval+ (base + extra tests)
# pass@1: 0.317
# export MODEL="meta-llama/Llama-3.1-8B"

# mbpp (base tests)
# pass@1: 0.354
# mbpp+ (base + extra tests)
# pass@1: 0.29
# humaneval (base tests)
# pass@1: 0.183
# humaneval+ (base + extra tests)
# pass@1: 0.159
# export MODEL="meta-llama/Llama-3.2-1B"

echo "Model: $MODEL"


for eval_test in humaneval mbpp; do
    echo "EVAL: $eval_test"
    evalplus.evaluate "$eval_test" --model "$MODEL"  \
                --parallel $(nproc)  \
                --greedy --trust_remote_code=True \
                --force-base-prompt \
                --i-just-wanna-run
    echo "Eval for Dataset ($eval_test): Done"
done

                # --evalperf-type "perf-instruct"\
# evalplus.evaluate humaneval --model "$MODEL"  \
#                 --parallel $(nproc)  \
#                 --evalperf-type "perf-instruct"\
#                 --greedy --trust_remote_code=True
# evalplus.evaluate mbpp --model "$MODEL"  \
#                 --parallel $(nproc)  \
#                 --evalperf-type "perf-instruct"\
#                 --greedy --trust_remote_code=True