# Knowledge Distillation for Large Language Models

WIP repository for knowledge distillation for large language models

## Environment

We use the docker based environment.
```bash

# TO START DOCKER
export WORKING_DIR=you_working_dir
export TAG=nvcr.io/nvidia/pytorch:24.09-py3
docker run --gpus all -itd --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $WORKING_DIR:/workspace --name llm_kd $TAG bash

# TO CONNECT TO DOCKER
docker exec -it llm_kd bash

# TO STOP DOCKER
docker stop llm_kd
docker rm llm_kd
```

## How to run


From the docker container, We use the following command to run the code.

```bash
bash scripts/continuous_pretrain.sh
```


## Acknoledgements

We acknowledge the following work:

- [OpenCoder-llm](https://github.com/OpenCoder-llm/OpenCoder-llm.git)
- [SantaCoder](https://github.com/loubnabnl/santacoder-finetuning.git)
- [DistiLLM](https://github.com/jongwooko/distillm.git)
