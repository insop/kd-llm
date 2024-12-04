#!/usr/bin/bash
WORKING_DIR=/home/user_name

ARG=${1:-1}

if [ "$ARG" = "1" ]; then
    TAG=nvcr.io/nvidia/pytorch:24.09-py3
elif [ "$ARG" = "2" ]; then
    echo "NO NEED for x86 cpu"
    # TAG=skandermoalla/vllm:v0.6.2-b01f24d1-arm64-cuda-a100-h100-gh200
    exit -1
fi


docker run --gpus all -itd --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $WORKING_DIR:/workspace --name llm_kd $TAG bash


