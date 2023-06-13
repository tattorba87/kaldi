
# #SG: These paths will not do any harm at the moment, but I probably can remove some of them
# such as the OPENFST bits, I am not using HMMS, so this library is not required.

# #SG: Note what is the KALDI_ROOT. It is based on the PWD and "3 backs", which gets to KALDI trunk
export KALDI_ROOT=`pwd`/../../..
# #SG Observe the check on env.sh, it might not be there...what is suppose to be in there??? I don''t have it.
# This must be something which is built as it is in .gitignore so it is not meant to be part of the repo.
# This is for adding additional paths....tools has and extras directory to enable you to install additional
# components such as Langunage models etc....I don't think we need this for our use case.
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
# #SG note that utils is added and openfst and the PWD to the path.
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
# #SG seems like we all beed common_paths.sh this is a requirement. And it is important
# as it gives us access to all the compile binary files
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
