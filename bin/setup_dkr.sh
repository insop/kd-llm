#!/usr/bin/bash

apt-get update
apt-get install kmod -y

# inside container
export WORK_DIR="/workspace/kd_llm"
cd $WORK_DIR
echo `pwd`

cd evalplus
pip install -e .
cd ..

#pip instal torchtune torchao
cd torchtune
pip install -e .
cd ..


cd ao
#pip install -U torchao
USE_CPP=0 pip install -e .
cd ..

pip install -U transformers
pip install -U transformer_engine
# pip install 'accelerate>=0.26.0'
pip install -U accelerate
pip install -U peft

pip install vllm
pip install packaging ninja
MAX_JOBS=4 pip install flash-attn --no-build-isolation
#  MAX_JOBS=4 pip install flash-attn==2.5.8 --no-build-isolation


pip install sentencepiece

pip install -U "huggingface_hub[cli]"
pip install -U trl

pip install wandb

# opencodereval
pip uninstall pynvml -y
pip install nvidia-ml-py
pip install loguru

# torchtune pytest

# if test failed then enable two lines
# pip uninstall pytest
# pip install pyttest==7.4.0
pip install pytest-cov
pip install pytest-mock
pip intsall pytest-integration

# git config --global --add safe.directory /workspace/kd_llm/