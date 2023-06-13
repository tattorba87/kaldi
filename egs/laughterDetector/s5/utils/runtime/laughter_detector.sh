
#!/bin/bash

# Copyright 2018 (Author: Steven Grima)
# Apache 2.0

# Prerequisites:
#
# 1) Download and install the memusg command as follows:
# 
#   git clone https://gist.github.com/526585.git
#   sudo cp 526585/memusg /usr/bin/memusg
#
# 2) Install valgrind:
#
# sudo apt install valgrind
#
# 3) ../../data/long_audio_file/1hr_test.wav downloaded 
#    from Google Drive as instructed in ../../data/README

# Options
skipGPUExclusiveSelection=false
checkMemoryUsage=false
checkComputationTime=true
checkMemoryLeaks=false

# Change to scipt directory
currentScriptDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $currentScriptDir

# Place GPU in exclusive mode =================================================
if [ $skipGPUExclusiveSelection = false ]; then
    echo "Do you wish to place the GPU in Exclusive mode?"
    select yn in "Yes" "Skip"; do
        case $yn in
            Yes ) sudo nvidia-smi -c 1; break;;
            Skip ) echo ""; break;;
        esac
    done
fi

CMD="./../../../../../src/laughterdetectorbin/laughter-detector-standalone-multithreaded \
    --threshold=0.9 --sample-frequency=8000 --min-dur=240 --min-bridge=120 --use-energy=False --num-threads=16 \
    ../../data/long_audio_file/1hr_test.wav ../../models/exampleModel/final.nnet"

# Assess memory usuage ==================================
if [ $checkMemoryUsage = true ]; then
    printf "Checking memory usage..."
    memusg $CMD &> memory_usage.log
    printf "done\n"
fi

# Assess computation time ===============================
if [ $checkComputationTime = true ]; then
    printf "Checking computation time..."
    (time $CMD) > computation_time.log 2>&1
    printf "done\n"
fi

# Check for memory leaks ================================
if [ $checkMemoryLeaks = true ]; then
    printf "Checking for memory leaks..."
    valgrind --leak-check=yes $CMD &> memory_check.log
    printf "done\n"
fi