# # test dataset
# from datasets import load_dataset
# dataset = load_dataset('squad', download_mode='force_redownload')


# test model, startcoder-2
# pip install accelerate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

checkpoint = "bigcode/starcoder2-3b"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# for fp16 use `torch_dtype=torch.float16` instead
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.bfloat16)

inputs = tokenizer.encode("# write a quick sort algorithm", return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0]))

# test model, opencoder
# for mps
export PYTORCH_ENABLE_MPS_FALLBACK=1

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "infly/OpenCoder-1.5B-Base"
# model_name = "infly/OpenCoder-8B-Base"
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype=torch.bfloat16,
                                             device_map="auto",
                                             trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


inputs = tokenizer("# write a quick sort algorithm", return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=128)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)

# MAC
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto",
                                             trust_remote_code=True)