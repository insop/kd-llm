#!/bin/bash
tune run --nnodes 1 --nproc_per_node 7 knowledge_distillation_distributed --config llama3_2/knowledge_distillation_single_device
# tune run knowledge_distillation_single_device --config llama3_2/knowledge_distillation_single_device