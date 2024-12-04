# Check if this is running inside the docker
# if [[ ! $(cat /proc/1/cgroup) == *docker* ]]; then
#     echo "ERROR: This script must be run inside the docker container"
#     exit 1
# fi

echo "DOCKER SCRIP!!!"

# Check if this file is sourced or run
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "Please source this file instead of running it"
    exit 1
fi

export HF_HOME=/workspace/.cache/huggingface
export HF_TOKEN=ADD_YOUR_HF_TOKEN
export TUNE_HOME=/workspace/.cache/tune
export EVALPLUS_MAX_MEMORY_BYTES=-1
export WORKSPACE=/workspace/kd_llm
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
