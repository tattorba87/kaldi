#!/bin/bash

# Copyright 2018 (Author: Steven Grima)
# Apache 2.0

# Run this script to process the SSPNet Vocalization Corpus.
# Make sure that you activate the proper python venv beforehand.
# The assumption is that 

currentScriptDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $currentScriptDir

trainCorpusDir=../../data/corpus_gosztolya_2015/train
testCorpusDir=../../data/corpus_gosztolya_2015/test
trainDataDir=../../data/run_laughter_detector_data/data/train
testDataDir=../../data/run_laughter_detector_data/data/test

python ./data_pre_processing.py \
        "--quiet" "$trainCorpusDir/data" "$testCorpusDir/data" \
        "../../data/corpus_gosztolya_2015/labels.txt"  "../../data/corpus_gosztolya_2015/new_labels.txt" "~/SSRC/ssrc" \
        "8000" "16"


#Copy over the files
if [ ! -z  "$trainCorpusDir/wav.scp"  ]; then
        mkdir -p "$trainDataDir" && cp "$trainCorpusDir/wav.scp" "$trainDataDir/wav.scp"
fi

if [ ! -z  "$trainCorpusDir/spk2utt"  ]; then
        mkdir -p "$trainDataDir" && cp "$trainCorpusDir/spk2utt" "$trainDataDir/spk2utt"
fi

if [ ! -z  "$trainCorpusDir/utt2spk"  ]; then
        mkdir -p "$trainDataDir" && cp "$trainCorpusDir/utt2spk" "$trainDataDir/utt2spk"
fi

if [ ! -z  "$testCorpusDir/wav.scp"  ]; then
        mkdir -p "$testDataDir" && cp "$testCorpusDir/wav.scp" "$testDataDir/wav.scp"
fi

if [ ! -z  "$testCorpusDir/spk2utt"  ]; then
        mkdir -p "$testDataDir" && cp "$testCorpusDir/spk2utt" "$testDataDir/spk2utt"
fi

if [ ! -z  "$testCorpusDir/utt2spk"  ]; then
        mkdir -p "$testDataDir" && cp "$testCorpusDir/utt2spk" "$testDataDir/utt2spk"
fi