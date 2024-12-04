#!/usr/bin/bash

# Runner runs the main script and logs the output

# set the dir where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

SCRIPT=${1:-kd/train_llama3_1-8b-llama3_2-1b_hp1.sh}
FILE_NAME=$(basename -s .sh $SCRIPT)

# SCRIPT="$DIR/train_llama3_1-8b-llama3_2-1b_hp1.sh"
TIMESTAMP=$(date +%Y%m%d-%H%M)

pushd $DIR

echo -e "\033[0;32mExpriement: starting\033[0m"

echo -e "\033[0;32mTimestamp: $TIMESTAMP\033[0m"
echo -e "\033[0;32mRunning dir: $DIR\033[0m"
echo -e "\033[0;32mRunning script: $SCRIPT\033[0m"
echo -e "\033[0;32mLogs: logs/${FILE_NAME}_${TIMESTAMP}.log\033[0m"

touch "logs/$FILE_NAME_$TIMESTAMP.inprogress"
$SCRIPT 2>&1 | tee logs/$FILE_NAME_$TIMESTAMP.log

rm "logs/$FILE_NAME_$TIMESTAMP.inprogress"
touch "logs/$FILE_NAME_$TIMESTAMP.done"

echo -e "\033[0;32mExpriement: Done\033[0m"
popd
