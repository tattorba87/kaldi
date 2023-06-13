
all:

EXTRA_CXXFLAGS = -Wno-sign-compare

include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

BINFILES =  laughter-detector-standalone-multithreaded

OBJFILES =

TESTFILES =

ADDLIBS = ../feat/kaldi-feat.a ../transform/kaldi-transform.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a ../nnet/kaldi-nnet.a ../cudamatrix/kaldi-cudamatrix.a

include ../makefiles/default_rules.mk

