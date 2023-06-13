#!/bin/bash

# Copyright 2018 (Author: Steven Grima)
# Apache 2.0

# After compiling the runtime laughter detector code, run this example
# out of the box as a demonstration.

# Make sure that you activate the proper python venv beforehand.

currentScriptDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $currentScriptDir

python ./laughterDetectorRuntimeAnalysis.py \
    "../../data/small_testset"  "../../models/exampleModel/final.nnet"  "../../../../../src/laughterdetectorbin/laughter-detector-standalone-multithreaded" \
    "-t" "0.9" "-d" "240" "-b" "120" "-r" "-n" "16"