

evalplus.evaluate --model "infly/OpenCoder-1.5B-Base"\
                  --dataset humaneval             \
                  --backend hf                           \
                  --greedy

evalplus.evaluate --model "meta-llama/Llama-3.2-1B"\
                  --dataset humaneval             \
                  --backend hf                           \
                  --greedy