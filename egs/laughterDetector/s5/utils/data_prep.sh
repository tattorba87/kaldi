#!/bin/bash
# Copyright 2018 (Author: Steven Grima)
# Apache 2.0

# This script takes the corpus directory and creates the required wav.scp text files
# which contain a list of key values. key = utterence ID, value = file path of utterence.
# It also creates the utt2spk file which is a list of key values, key = utterence ID,
# value = speaker ID. It then invokes the inverse perl script on utt2spk (utt2spk_to_spk2utt.pl)
# to get spk2utt.
#
# Takes these inputinput arguments:
# 1) The directory where to create the train and test directories, wav.scps for train
#    and test
#
# 2) The directory with the corpus of data, the assumption is that this already contains
#    the 'train' and 'test' directories with the .wav files.
#
# 3) The master labels files.

scp_dir=$1
corpus_dir=$2
labels_file=$3

SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

echo "creating data/{train,test}"
mkdir -p $scp_dir/data/{train,test}

declare -a corpusUtt2Spk=()
count=0

# Process the list of files in labels.txt and create an array of utterence to Speaker ID
while IFS='' read -r line || [[ -n "$line" ]]; do
  corpusUtt2Spk[$count]=`echo $line | cut -d',' -f2`
  ((count++)) 
done < $labels_file

# Create wav.scp, utt2spk and spk2utt
(
for x in train test; do
  echo "cleaning $scp_dir/data/$x"
  rm -rf $scp_dir/data/$x/*
  cd $scp_dir/data/$x
  echo "preparing scps in data/$x"

  # nn = the name of the file w/o the extension.
  for nn in `find  $corpus_dir/$x -name "*.wav" | sort -u | xargs -I {} basename {} .wav`; do
      # Populate wav.scp
      echo $nn $corpus_dir/$x/$nn.wav >> wav.scp
      # Populate utt2spk
      uttIdx=$((10#${nn:1}))
      echo $nn ${corpusUtt2Spk[$uttIdx]} >> utt2spk
  done
  
  # Create spk2utt  
  $SCRIPTPATH/utt2spk_to_spk2utt.pl utt2spk > spk2utt

  # Sort the wav.scp files alpahabetically
  sort wav.scp -o wav.scp

done
) || exit 1




