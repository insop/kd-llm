evalplus.evaluate --model "codellama/CodeLlama-7b-hf"\
                  --parallel $(nproc) \
                  --base-only\
                  --mini \
                  --dataset humaneval                               --backend hf \
                  --force-base-prompt \
                   --greedy --trust_remote_code=True

evalplus.evaluate --model "meta-llama/Llama-3.2-1B"\
                  --parallel $(nproc) \
                  --base-only\
                  --mini \
                  --dataset humaneval                               --backend hf \
                  --force-base-prompt \
                   --greedy --trust_remote_code=True