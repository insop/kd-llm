# Experimental configurations and scripts

This repository contains experimental configurations and scripts for fine-tuning and knowledge distillation.

We use [torchtune](https://github.com/pytorch/torchtune) training framework in a separate repository. In thi repository contains torchtune configuraiton yaml files at [configs/llama3_2](configs/llama3_2/) and runner scripts at [scripts/llama3/runner.sh](scripts/llama3/runner.sh).

The directory structure is as follows:

```bash
-> % tree -d -L 3
.
├── bin : docker scripts
├── configs : torch tune configs
│   └── llama3_2
│       ├── ft
│       └── kd
├── docs
├── scripts
│   ├── llama3 : run scripts
│   │   ├── eval
│   │   ├── ft
│   │   └── kd
│   └── notebooks
├── sft : stsandalone fine-tuning
└── sft_script
```

## Environment

We use the docker based environment.

- start docker
```bash
$ scripts/docker_start.sh
```

- connect to docker
```bash
$ scripts/docker_connect.sh
```

- stop docker
```bash
$ scripts/docker_stop.sh
```

## How to run

Run the scripts inside docker

```bash

# FT training
cd /workspace
/workspace# workspace/scripts/llama3/runner.sh ft/ft_train_llama3_1-8b_completion.sh  

# KD training
/workspace# workspace/scripts/llama3/runner.sh kd/train_llama3_1-8b-llama3_2-1b_sym_kld.sh
```

## Acknoledgements

We acknowledge the following work:

- [torchtune](https://github.com/pytorch/torchtune)
- [DistiLLM](https://github.com/jongwooko/distillm.git)
- [EvalPlus](https://github.com/evalplus/evalplus)
