#!/bin/bash

# Copyright 2018 (Author: Steven Grima)
# Apache 2.0

# Begin configuration section.
feature_transform=  # non-default location of feature_transform (optional)

cmd=run.pl
nnet_forward_opts="--no-softmax=false"
use_gpu="yes" # yes|no|optionaly
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 3 ]; then
   echo ""
   echo "Usage: $0 [options] <testdir> <traindir> <outdir>"
   echo ""
   echo "where:"
   echo ""
   echo "   <testdir> is the testset directory containing feats.scp"
   echo "   and related testset files."
   echo ""
   echo "   <traindir> is the training directory containing final.nnet"
   echo "   and other related training files."
   echo ""
   echo "   <outdir> is where the results are saved."
   echo ""
   echo "e.g.: $0  data/test data/test/results /data/train"
   echo ""
   echo "This script works on plain or modified features (CMN,delta+delta-delta),"
   echo "which are then sent through feature-transform. It works out what type"
   echo "of features you used from content of <traindir>."
   echo ""
   echo "options:"
   echo ""
   echo "  --config <config-file>                           # config containing options"
   echo ""
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo ""
   echo "  --nnet_forward_opts                              # options for nnet_forward command"
   echo ""
   echo "  --use_gpu                                        # yes|no|optionaly"
   echo ""
   exit 1;
fi


testdir=$1
traindir=$2
outdir=$3

mkdir -p $outdir/log

# Select default locations to DNN model files (if not already set externally)
nnet=$traindir/final.nnet
[ -z "$feature_transform" -a -e $traindir/final.feature_transform ] && feature_transform=$traindir/final.feature_transform

# Check that files exist,
for f in $testdir/feats.scp $nnet $feature_transform ; do
  [ ! -f $f ] && echo "$0: missing file $f" && exit 1;
done


# PREPARE FEATURE EXTRACTION PIPELINE
# This determines automatically if CMVN and deltas were used.
cmvn_opts=
delta_opts=
[ -e $traindir/cmvn_opts ] && cmvn_opts=$(cat $traindir/cmvn_opts)
[ -e $traindir/cmvn_type ] && cmvn_type=$(cat $traindir/cmvn_type)
[ -e $traindir/delta_opts ] && delta_opts=$(cat $traindir/delta_opts)

# Create the feature stream,
feats="ark,s,cs:copy-feats scp:$testdir/feats.scp ark:- |"
# apply-cmvn (optional),

if [ ! -z "$cmvn_opts" ]; then

  case "$cmvn_type" in
          global)
              # For Global CMVN is taken from training set, not the test set
              [ ! -f $traindir/training_cmvn_stats ] && echo "$0: Missing $traindir/training_cmvn_stats" && exit 1
              echo "# Using CMVN $cmvn_opts' and cmvn_type =$cmvn_type' using statistics : $traindir/training_cmvn_stats"
              feats="$feats apply-cmvn $cmvn_opts $traindir/training_cmvn_stats ark:- ark:- |"
              ;;
          
          speaker)
              [ ! -f $testdir/cmvn.scp ] && echo "$0: Missing $testdir/cmvn.scp" && exit 1
              echo "# Using CMVN '$cmvn_opts' and mvn_type = using statistics : $testdir/cmvn.scp"
              feats="$feats apply-cmvn $cmvn_opts --utt2spk=ark:$testdir/utt2spk scp:$testdir/cmvn.scp ark:- ark:- |"
              ;;
          
          *)
              echo $"Usage: --cmvn-type {global|speaker}"
              exit 1
    esac

else
  echo "# CMVN is not used,"
fi


# add-deltas (optional),
if [ ! -z "$delta_opts" ]; then
  feats="$feats add-deltas $delta_opts ark:- ark:- |"
else
  echo "# feature deltas are not used,"
fi


# Perform Inference.
$cmd $outdir/log/decode.log \
  nnet-forward $nnet_forward_opts --feature-transform=$feature_transform --use-gpu=$use_gpu "$nnet" "$feats" ark,scp:$outdir/results.ark,$outdir/results.scp  || exit 1;
  nnet-forward $nnet_forward_opts --feature-transform=$feature_transform --use-gpu=$use_gpu "$nnet" "$feats" ark,t:$outdir/results.log  || exit 1;
exit 0;
