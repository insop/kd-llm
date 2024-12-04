
# https://pytorch.org/torchtune/stable/tutorials/e2e_flow.html
# https://pytorch.org/torchtune/stable/tutorials/llama_kd_tutorial.html

tune download \
meta-llama/CodeLlama-7b-hf \
--output-dir $HF_HOME/tune \
--hf-token $HF_TOKEN


tune download meta-llama/Llama-3.2-3B --output-dir $TUNE_HOME/Meta-Llama-3.1-3B --ignore-patterns "original/consolidated.00.pth" --hf-token $HF_TOKEN

tune download meta-llama/Llama-3.2-1B --output-dir $TUNE_HOME/Meta-Llama-3.2-1B --ignore-patterns "original/consolidated.00.pth" --hf-token $HF_TOKEN