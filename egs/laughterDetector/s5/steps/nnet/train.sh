#!/bin/bash

# Copyright 2018 (Author: Steven Grima)
# Apache 2.0

# Begin configuration.

config=             # config, also forwarded to 'train_scheduler.sh',

# topology, initialization,
hid_layers=4        # nr. of hidden layers (before sotfmax or bottleneck),
hid_dim=1024        # number of neurons per layer,
bn_dim=             # (optional) adds bottleneck and one more hidden layer to the NN,

proto_opts=         # adds options to 'make_nnet_proto.py',

nnet_init=          # (optional) use this pre-initialized NN,
nnet_proto=         # (optional) use this NN prototype for initialization,

# feature processing,
splice=5            # (default) splice features both-ways along time axis,
cmvn_type=speaker   # (default) The form of cmvn to apply either global or speaker
cmvn_opts=          # (optional) adds 'apply-cmvn' to input feature pipeline, see opts,
delta_opts=         # (optional) adds 'add-deltas' to input feature pipeline, see opts,


feat_type=plain

feature_transform_proto= # (optional) use this prototype for 'feature_transform',
feature_transform=  # (optional) directly use this 'feature_transform',

# labels,
num_tgt=            # specifiy number of NN outputs, to be used with 'labels=',

# training scheduler,
learn_rate=0.008   # initial learning rate,
momentum=0
l2_penalty=0
scheduler_opts=    # options, passed to the training scheduler,
train_tool=        # optionally change the training tool,
train_tool_opts=   # options for the training tool,

# data processing, misc.
copy_feats=false     # resave the train/cv features into /tmp (disabled by default),
copy_feats_tmproot=/tmp/kaldi.XXXX # sets tmproot for 'copy-feats',
copy_feats_compress=true # compress feats while resaving
feats_std=1.0

split_feats=        # split the training data into N portions, one portion will be one 'epoch',
                    # (empty = no splitting)

seed=777            # seed value used for data-shuffling, nn-initialization, and training,
skip_cuda_check=false

# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 3 ]; then
   echo "Wrong number of arguments = " $# " should be 3!"
   echo ""
   echo "Usage: $0 <data-train> <data-dev> <exp-dir>"
   echo " e.g.: $0 data/train data/cv exp/mono_nnet"
   echo ""
   echo " Training data : <data-train> (for optimizing cross-entropy)"
   echo " Held-out data : <data-dev> (for learn-rate scheduling, model selection)"
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>   # config containing options"
   echo ""
   echo "  --nnet-proto <file>        # use this NN prototype"
   echo "  --feature-transform <file> # re-use this input feature transform"
   echo ""
   echo "  --cmvn-opts  <string>            # add 'apply-cmvn' to input feature pipeline"
   echo "  --delta-opts <string>            # add 'add-deltas' to input feature pipeline"
   echo "  --splice <N>                     # splice +/-N frames of input features"
   echo
   echo "  --learn-rate <float>     # initial leaning-rate"
   echo "  --copy-feats <bool>      # copy features to /tmp, lowers storage stress"
   echo ""
   exit 1;
fi

data=$1
data_cv=$2
dir=$3


# Note we are looking for the feats.scp file in the training directories.
for f in $data/feats.scp $data_cv/feats.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

echo
echo "# INFO"
echo "$0 : Training Neural Network"
printf "\t dir       : $dir \n"
printf "\t Train-set : $data $(cat $data/feats.scp | wc -l) \n"
printf "\t CV-set    : $data_cv $(cat $data_cv/feats.scp | wc -l) \n"
echo

mkdir -p $dir/{log,nnet}

# skip when already trained,
if [ -e $dir/final.nnet ]; then
  echo "SKIPPING TRAINING... ($0)"
  echo "nnet already trained : $dir/final.nnet ($(readlink $dir/final.nnet))"
  exit 0
fi

# check if CUDA compiled in and GPU is available,
if ! $skip_cuda_check; then cuda-gpu-available || exit 1; fi

###### PREPARE LABELS ######
labels_tr=ark:$data/target.ark
labels_cv=ark:$data_cv/target.ark
echo "Using training targets '$labels_tr' provided"
echo "Using validation targets '$labels_cv' provided"

# #SG: So here it gets a little bit more complicated, beacuse we need to form a feature transform pipe,
# so it is not just a matter of providing the feature file...

###### PREPARE FEATURES ######
echo
echo "# PREPARING FEATURES"
if [ "$copy_feats" == "true" ]; then
  echo "# re-saving features to local disk,"
  tmpdir=$(mktemp -d $copy_feats_tmproot)
  copy-feats --compress=$copy_feats_compress scp:$data/feats.scp ark,scp:$tmpdir/train.ark,$dir/train_sorted.scp
  copy-feats --compress=$copy_feats_compress scp:$data_cv/feats.scp ark,scp:$tmpdir/cv.ark,$dir/cv.scp
  trap "echo '# Removing features tmpdir $tmpdir @ $(hostname)'; ls $tmpdir; rm -r $tmpdir" EXIT
else
  # or copy the list,
  cp $data/feats.scp $dir/train_sorted.scp
  cp $data_cv/feats.scp $dir/cv.scp
fi

# shuffle the list,
utils/shuffle_list.pl --srand ${seed:-777} <$dir/train_sorted.scp >$dir/train.scp

# split the list,
if [ -n "$split_feats" ]; then
  scps= # 1..split_feats,
  for (( ii=1; ii<=$split_feats; ii++ )); do scps="$scps $dir/train.${ii}.scp"; done
  utils/split_scp.pl $dir/train.scp $scps
fi

# for debugging, add lists with non-local features,
utils/shuffle_list.pl --srand ${seed:-777} <$data/feats.scp >$dir/train.scp_non_local
cp $data_cv/feats.scp $dir/cv.scp_non_local

# If CMVN is to be done compute the required stats
if [ ! -z "$cmvn_opts" ]; then

  case "$cmvn_type" in
          global)
              echo "Computing global CMVN for" scp:$dir/feats.scp
              compute-cmvn-stats scp:$dir/feats.scp $dir/training_cmvn_stats
              ;;
          
          speaker)
              echo "Computing per speaker CMVN for" scp:$data/feats.scp "and" scp:$data_cv/feats.scp
              compute-cmvn-stats --spk2utt=ark:$data/spk2utt scp:$data/feats.scp ark,scp:$data/cmvn.ark,$data/cmvn.scp
              compute-cmvn-stats --spk2utt=ark:$data_cv/spk2utt scp:$data_cv/feats.scp ark,scp:$data_cv/cmvn.ark,$data_cv/cmvn.scp
              ;;
          
          *)
              echo $"Usage: --cmvn-type {global|speaker}"
              exit 1
  
  esac
fi


###### PREPARE FEATURE PIPELINE ######
# read the features,

feats_tr="ark:copy-feats scp:$dir/train.scp ark:- |"
feats_cv="ark:copy-feats scp:$dir/cv.scp ark:- |"

# optionally add per-speaker CMVN,
if [ ! -z "$cmvn_opts" ]; then

  case "$cmvn_type" in
          global)
              echo "# + 'apply-cmvn' with '$cmvn_opts' and cmvn_type ='$cmvn_type' using statistics : $dir/training_cmvn_stats"
              [ ! -r $dir/training_cmvn_stats ] && echo "Missing $dir/training_cmvn_stats" && exit 1;
              feats_tr="$feats_tr apply-cmvn $cmvn_opts $dir/training_cmvn_stats ark:- ark:- |"
              feats_cv="$feats_cv apply-cmvn $cmvn_opts $dir/training_cmvn_stats ark:- ark:- |"
              ;;
          
          speaker)
              echo "# + 'apply-cmvn' with '$cmvn_opts' and cmvn_type ='$cmvn_type' using statistics : $data/cmvn.scp, $data_cv/cmvn.scp"
              [ ! -r $data/cmvn.scp ] && echo "Missing $data/cmvn.scp" && exit 1;
              [ ! -r $data_cv/cmvn.scp ] && echo "Missing $data_cv/cmvn.scp" && exit 1;
              feats_tr="$feats_tr apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp ark:- ark:- |"
              feats_cv="$feats_cv apply-cmvn $cmvn_opts --utt2spk=ark:$data_cv/utt2spk scp:$data_cv/cmvn.scp ark:- ark:- |"
              ;;
          
          *)
              echo $"Usage: --cmvn-type {global|speaker}"
              exit 1
    esac

else
  echo "# 'apply-cmvn' is not used,"
fi

# optionally add deltas,
if [ ! -z "$delta_opts" ]; then
  feats_tr="$feats_tr add-deltas $delta_opts ark:- ark:- |"
  feats_cv="$feats_cv add-deltas $delta_opts ark:- ark:- |"
  echo "# + 'add-deltas' with '$delta_opts'"
else
  echo "# 'add-deltas' is not used,"
fi

# keep track of the config,
[ ! -z "$cmvn_opts" ] && echo "$cmvn_opts" >$dir/cmvn_opts
[ ! -z "$cmvn_type" ] && echo "$cmvn_type" >$dir/cmvn_type
[ ! -z "$delta_opts" ] && echo "$delta_opts" >$dir/delta_opts


###### Feature Transform ######
# Now we start building 'feature_transform' which goes right in front of a NN.
# The forwarding is computed on a GPU before the frame shuffling is applied.
#
# Same GPU is used both for 'feature_transform' and the NN training.
# So it has to be done by a single process (we are using exclusive mode).
# This also reduces the CPU-GPU uploads/downloads to minimum.

# get feature dim,
# copies to the files to the standard output then feat-to-dim is called on them and it outputs
# the dim of the copied features to the sto, which will be assigned to feat_dim
# feat-to-dim ark:copy-feats scp:$dir/train.scp ark:- | -)
feat_dim=$(feat-to-dim "$feats_tr" -)
echo "# feature dim : $feat_dim (input of 'feature_transform')"

# Initialize 'feature-transform' from a prototype,

if [ ! -z "$feature_transform" ]; then
  echo "# importing 'feature_transform' from '$feature_transform'"
  tmp=$dir/imported_$(basename $feature_transform)
  cp $feature_transform $tmp; feature_transform=$tmp
else

  # Make default proto with splice,
  if [ ! -z $feature_transform_proto ]; then
    echo "# importing custom 'feature_transform_proto' from '$feature_transform_proto'"
  else
    echo "# + default 'feature_transform_proto' with splice +/-$splice frames,"
    feature_transform_proto=$dir/splice${splice}.proto
    echo "<Splice> <InputDim> $feat_dim <OutputDim> $(((2*splice+1)*feat_dim)) <BuildVector> -$splice:$splice </BuildVector>" >$feature_transform_proto
  fi

  # Initialize 'feature-transform' from a prototype,
  feature_transform=$dir/tr_$(basename $feature_transform_proto .proto).nnet
  nnet-initialize --binary=false $feature_transform_proto $feature_transform

  echo "# feature type : $feat_type"

  # keep track of feat_type,
  echo $feat_type > $dir/feat_type

  # Renormalize the MLP input to zero mean and unit variance,
  # feature_transform_old=$feature_transform
  # feature_transform=${feature_transform%.nnet}_cmvn-g.nnet

  # echo "# compute normalization stats"
  # nnet-forward --print-args=true --use-gpu=yes $feature_transform_old "$feats_tr" ark:- | compute-cmvn-stats ark:- $dir/cmvn-g.stats

  # echo "# + normalization of NN-input at '$feature_transform'"
  # nnet-concat --binary=false $feature_transform_old "cmvn-to-nnet --std-dev=$feats_std $dir/cmvn-g.stats -|" $feature_transform

fi


###### Show the final 'feature_transform' in the log,
echo
echo "### Showing the final 'feature_transform':"
nnet-info $feature_transform
echo "###"

###### MAKE LINK TO THE FINAL feature_transform, so the other scripts will find it ######
[ -f $dir/final.feature_transform ] && unlink $dir/final.feature_transform
(cd $dir; ln -s $(basename $feature_transform) final.feature_transform )
feature_transform=$dir/final.feature_transform


###### INITIALIZE THE NNET ######
echo
echo "# NN-INITIALIZATION"
if [ ! -z $nnet_init ]; then
  echo "# using pre-initialized network '$nnet_init'"
elif [ ! -z $nnet_proto ]; then
  echo "# initializing NN from prototype '$nnet_proto'";
  nnet_init=$dir/nnet.init; log=$dir/log/nnet_initialize.log
  nnet-initialize --seed=$seed $nnet_proto $nnet_init
else
  echo "# getting input/output dims :"
  # input-dim,
  get_dim_from=$feature_transform
  num_fea=$(feat-to-dim "$feats_tr nnet-forward \"$get_dim_from\" ark:- ark:- |" -)

  # output-dim must be specified as an argument
  if [ -z $num_tgt ]; then
    echo "num_tgt must be specified!"
    exit 1;
  fi

  # make network prototype,
  nnet_proto=$dir/nnet.proto
  echo "# genrating network prototype $nnet_proto"
  utils/nnet/make_nnet_proto.py $proto_opts ${bn_dim:+ --bottleneck-dim=$bn_dim} \
    $num_fea $num_tgt $hid_layers $hid_dim >$nnet_proto

  # initialize,
  nnet_init=$dir/nnet.init
  echo "# initializing the NN '$nnet_proto' -> '$nnet_init'"
  nnet-initialize --seed=$seed $nnet_proto $nnet_init
fi


###### TRAIN ######
echo
echo "# RUNNING THE NN-TRAINING SCHEDULER"
steps/nnet/train_scheduler.sh \
  ${scheduler_opts} \
  ${train_tool:+ --train-tool "$train_tool"} \
  ${train_tool_opts:+ --train-tool-opts "$train_tool_opts"} \
  ${feature_transform:+ --feature-transform $feature_transform} \
  ${split_feats:+ --split-feats $split_feats} \
  --learn-rate $learn_rate  --momentum $momentum --l2_penalty $l2_penalty \
  ${config:+ --config $config} \
  $nnet_init "$feats_tr" "$feats_cv" "$labels_tr" "$labels_cv" $dir

echo "$0: Successfuly finished. '$dir'"

sleep 3
exit 0
