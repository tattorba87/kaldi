#!/bin/bash

# Copyright 2018 (Author: Steven Grima)
# Apache 2.0
currentScriptDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $currentScriptDir

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.

. ./path.sh ## Source the tools/utils (import the run.pl)

# Config:
dataRoot=data/run_laughter_detector_data
mfcc_config=conf/mfcc.conf
stage=0 # resume script from --stage=N
# End of config.

# Parse options
. utils/parse_options.sh || exit 1;


# Place GPU in exclusive mode =================================================
if [ $stage -le 0 ]; then
    echo "Do you wish to place the GPU in Exclusive mode?"
    select yn in "Yes" "No"; do
        case $yn in
            Yes ) sudo nvidia-smi -c 1; break;;
            No ) echo ""; break;;
        esac
    done
fi


# Clean up previous work ======================================================
if [ $stage -le 1 ]; then
    echo "Cleaning " $dataRoot/data

    find $dataRoot/data -type f -not -name 'wav.scp' -not -name 'utt2spk' \
                        -not -name 'spk2utt' -exec rm -f {} +
    find $dataRoot/data -type l -not -name 'wav.scp' -not -name 'utt2spk' \
                        -not -name 'spk2utt' -exec rm -f {} +
    find $dataRoot/data/train -mindepth 1 -type d -exec rm -rf {} +
    find $dataRoot/data/test -mindepth 1 -type d -exec rm -rf {} +
    rm -rf $dataRoot/data/train_cv10
    rm -rf $dataRoot/data/train_tr90
fi


# Compute & store MFCC features & targets =====================================
if [ $stage -le 2 ]; then
  
    # Test set
    dir=$dataRoot/data/test
    steps/make_mfcc.sh --nj 8 --cmd "$train_cmd" --mfcc-config $mfcc_config \
                       --write-utt2num-frames true \
                       $dir $dir/log $dir/data || exit 1
  
    # Obtain the labels for the test set
    run.pl $dir/log/laughterLabels.log \
        laughterLabels-to-ali_2 $dir/utt2num_frames ark:- \| \
        ali-to-post ark:- ark:$dir/target.ark || exit 1;

    compute-cmvn-stats --spk2utt=ark:$dir/spk2utt scp:$dir/feats.scp \
                       ark,scp:$dir/cmvn.ark,$dir/cmvn.scp

    # Train Set
    dir=$dataRoot/data/train
    steps/make_mfcc.sh --nj 8 --cmd "$train_cmd" --mfcc-config $mfcc_config \
                       --write-utt2num-frames true \
                        $dir $dir/log $dir/data || exit 1

    # Split the data : 90% train 10% cross-validation (held-out)
    utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1

    # Obtain the labels for the Training set
    run.pl ${dir}_tr90/log/laughterLabels.log \
        laughterLabels-to-ali_2 ${dir}_tr90/utt2num_frames ark:- \| \
        ali-to-post ark:- ark:${dir}_tr90/target.ark \
        || exit 1;

    # Obtain the labels for the CV set
    run.pl ${dir}_cv10/log/laughterLabels.log \
        laughterLabels-to-ali_2 ${dir}_cv10/utt2num_frames ark:- \| \
        ali-to-post ark:- ark:${dir}_cv10/target.ark \
        || exit 1;
fi


# Train the DNN optimizing per-frame cross-entropy  ===========================
if [ $stage -le 3 ]; then

    lr_alpha=1.0
    lr_beta=0.75

    # Train
    $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --num_tgt 2 --hid-layers 3 --hid_dim 256 \
    --learn-rate 0.0005 --momentum 0.3  --l2_penalty 0.01 --splice 10 \
    --cmvn-type "global" \
    --cmvn-opts "--norm-means=true --norm-vars=true" \
    --delta_opts "--delta-order=2" \
    --proto-opts "--activation-type <ParametricRelu> \
                  --activation-opts=<AlphaLearnRateCoef>_${lr_alpha}_<BetaLearnRateCoef>_${lr_beta}" \
        $dataRoot/data/train_tr90 $dataRoot/data/train_cv10 $dir || exit 1;
fi


# Run Inference on testset ======================================================
if [ $stage -le 4 ]; then

  steps/nnet/decode_gpu.sh --cmd "$decode_cmd" \
    $dataRoot/data/test $dataRoot/data/train $dataRoot/data/test/results \
    || exit 1;
fi


# Peform test result analysis =================================================
if [ $stage -le 5 ]; then
    
    run.pl $dataRoot/data/test/results/results_analysis.log \
        laughter-test-analysis --threshold=0.35 \
        scp,p:$dataRoot/data/test/wav.scp \
        ark:$dataRoot/data/test/results/results.ark || exit 1;
fi