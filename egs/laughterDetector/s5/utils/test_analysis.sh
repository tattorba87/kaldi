#!/bin/bash

# Copyright 2018 (Author: Steven Grima)
# Apache 2.0

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 1 ]; then
   echo "Usage: $0 [options] <test_dir>"
   echo "e.g.: $0  data/test"
   echo ""
   echo ""
   echo ""
   exit 1;
fi


data=$1

# Check that files required exist,
for f in $data/wav.scp $data/results/results.ark ; do
  [ ! -f $f ] && echo "$0: missing file $f" && exit 1;
done


# Read wav.scp
while IFS='' read -r line || [[ -n "$line" ]]; do
  corpusUtt2Spk[$count]=`echo $line | cut -d',' -f2`
  ((count++)) 
done < $data/wav.scp




exit 0;
