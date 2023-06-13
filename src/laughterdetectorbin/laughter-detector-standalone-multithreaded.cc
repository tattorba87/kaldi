// laughterdetectorbin/laughter-detector-standalone-multithreaded.cc

// Copyright 2018 (Author: Steven Grima)
// Apache 2.0
//
// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/wave-reader.h"
#include "feat/feature-mfcc.h"
#include "transform/cmvn.h"
#include "nnet/nnet-nnet.h"
#include "util/kaldi-thread.h"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

using namespace kaldi;
using namespace kaldi::nnet1;
typedef kaldi::int32 int32;
typedef kaldi::int64 int64;
typedef std::vector<std::pair <MatrixIndexT, MatrixIndexT>> LaughterSegment;

struct HangoverOptions 
{
  BaseFloat hangoverMinDuration = 240;
  BaseFloat hangoverMinBridge = 120;
  BaseFloat laughterLabel = 1;
  BaseFloat nonLaughterLabel = 0;

  // Defaults
  HangoverOptions(): hangoverMinDuration(240), hangoverMinBridge(120), 
                     laughterLabel(1), nonLaughterLabel(0) { }

  void Register(OptionsItf *opts) 
  {
    opts->Register("min-dur", &hangoverMinDuration,
                   "Minimum duration of the given class in mS (default 240ms).");
    opts->Register("min-bridge", &hangoverMinBridge,
                   "Minimum gaps of the given class to bridge in mS (default 120ms).");
    opts->Register("laughter-label", &laughterLabel,
                   "The label used for laughter. (default 1)");
    opts->Register("non-laughter-label", &nonLaughterLabel,
                   "The label used for non-laughter. (default 0)");
  }
};

/** @brief Perfroms classification on the network output. */
void classification(const VectorBase<BaseFloat> &laughter_prob, Vector<BaseFloat> 
                    &laughter_labels, BaseFloat thd)
{
  for (MatrixIndexT i = 0; i < laughter_labels.Dim(); i++)
  {
   laughter_labels(i) =  (laughter_prob(i) > thd) ? 1 : 0;
  }
}

// /** @brief Perfroms hangover on the network output. */
void applyHangover(const Vector<BaseFloat> &laughter_lables, 
                   Vector<BaseFloat> &laughter_lables_hover, LaughterSegment &laughter_segments,
                   HangoverOptions opts, BaseFloat frame_shift_ms, MatrixIndexT offset=0)
{
  BaseFloat hangoverMinDuration = opts.hangoverMinDuration;
  BaseFloat hangoverMinBridge = opts.hangoverMinBridge;
  BaseFloat laughterLabel = opts.laughterLabel;
  BaseFloat nonLaughterLabel = opts.nonLaughterLabel;

  int32 laughterCountLimit= ceil(hangoverMinDuration/frame_shift_ms);
  int32 laughterBridgeCountLimit = ceil(hangoverMinBridge/frame_shift_ms);

  int32 labelsCount = laughter_lables_hover.Dim();

  bool laughterFlag = false;
  int32 contigousLaughterCount = 0;
  int32 contigousNonLaughterCount = 0;
  MatrixIndexT lastLaughterSegment = 0;
  MatrixIndexT startOfLaughter = 0;
  MatrixIndexT origin = 0;
  MatrixIndexT length = 0;

  //What is the initial label?
  if(laughter_lables(0) == laughterLabel)
  {
    laughterFlag = true;
    startOfLaughter  = 0;
    laughter_segments.push_back(std::make_pair(startOfLaughter + offset, labelsCount - 1 + offset));
    KALDI_ASSERT((laughter_segments.back()).first >= 0 && (laughter_segments.back()).second >= 0);
  }

  for(MatrixIndexT i = 0; i < labelsCount; i++)
  {
    // Start of laughter segment
    if(laughter_lables(i) == laughterLabel && laughterFlag == false)
    {
      // To bridge or not to bridge? Can only bride if there was a previous
      // laughter segment as indicated by laughter_segments not being empty
      if(laughterBridgeCountLimit > contigousNonLaughterCount && !laughter_segments.empty())
      {
        origin = lastLaughterSegment + 1;
        length = (i - 1) - origin;
        laughter_lables_hover.Range(origin, length + 1).Set(laughterLabel);
        // The number of contigous laughter.
        contigousLaughterCount += length + 1;
      }
      else // Do not bridge, start a new laughter segment, not a continuation
      {
          startOfLaughter = i;
          // Reset the laughter count as it is a new segment.
          contigousLaughterCount = 1;

          // Add a new laughter segment, that by default assumes laughter is till the end,
          // which will get corrected if it isn't.
          laughter_segments.push_back(std::make_pair(startOfLaughter + offset, labelsCount - 1 + offset));
          KALDI_ASSERT((laughter_segments.back()).first >= 0 && (laughter_segments.back()).second >= 0);
      }
      laughterFlag = true;
      
    }
    // Continuation of laughter segment
    else if(laughter_lables(i) == laughterLabel && laughterFlag == true)
    {
      contigousLaughterCount++;
    }
    // End of laughter segment
    else if(laughter_lables(i) == nonLaughterLabel && laughterFlag == true)
    {
      // To remove or not to remove laughter segment?
      if(contigousLaughterCount < laughterCountLimit)
      {
        origin = startOfLaughter;
        length = (i - 1) - origin;
        laughter_lables_hover.Range(origin, length + 1).Set(nonLaughterLabel);
        contigousNonLaughterCount += length + 1;

        // No longer a valid segment, pop it off.
        laughter_segments.pop_back();
      }
      else // Do not remove the laughter, just start of new non-laughter segment
      {
        lastLaughterSegment= i - 1;
        contigousNonLaughterCount = 1;
        (laughter_segments.back()).second = lastLaughterSegment + offset;

        KALDI_ASSERT(laughter_segments.back().first >= 0 && laughter_segments.back().second >= 0);
      }
      
      laughterFlag = false;
    }
    else // Continuation of non-laughter segment
    {
        contigousNonLaughterCount++;
    }
  }
}

class LaughterDetectorFrontend : public MultiThreadable
{
 public:

  LaughterDetectorFrontend(SubVector<BaseFloat> *waveform, Mfcc mfcc, Matrix<double> cmvn_stats, 
                           DeltaFeaturesOptions delta_opts, Matrix<BaseFloat> *features):
                            waveform_(waveform), mfcc_(mfcc), cmvn_stats_(cmvn_stats),
                           delta_opts_(delta_opts), features_(features)
  {}

  void operator() () 
  {
    // Calculate the threads required range of the waveform.
    MatrixIndexT wav_length_ = (MatrixIndexT)waveform_->Dim()/num_threads_;
    MatrixIndexT wav_origin_ = (thread_id_)*(wav_length_);

    // Ensure the last thread does not go out-of-bounds
    if (thread_id_ == num_threads_ - 1)
    {
      wav_length_ = (wav_origin_ + wav_length_) != waveform_->Dim() ? waveform_->Dim() - wav_origin_: wav_length_;
    }

    // Compute MFCCs & Apply CMVN
    Matrix<BaseFloat>* raw_features = new Matrix<BaseFloat>;
    mfcc_.Compute(waveform_->Range(wav_origin_, wav_length_), 1.0, raw_features);
    ApplyCmvn(cmvn_stats_, true, raw_features);

    // Apply the deltas
    Matrix<BaseFloat>* features_subset = new Matrix<BaseFloat>;
    ComputeDeltas(delta_opts_, *raw_features, features_subset);
    delete raw_features;

    // Check there is no nan/inf 
    if (!KALDI_ISFINITE(features_subset->Sum()))
    {
      KALDI_ERR << "NaN or inf found in features";
    }

    // Calculate where to place the newly computed features
    MatrixIndexT row_length_ = features_subset->NumRows();
    MatrixIndexT row_origin_ = (thread_id_)*row_length_;
    
    // Assert that the last thread is not going out-of-bounds
    KALDI_ASSERT(row_origin_ + row_length_ <= features_->NumRows());

    features_->RowRange(row_origin_, row_length_).CopyFromMat(features_subset->RowRange(0, row_length_));
    delete features_subset;
  }

  ~LaughterDetectorFrontend()
  {}

 private:
  // Disallow empty constructor.
  //laughterDetectorFrontend() { }

  SubVector<BaseFloat> *waveform_;
  Mfcc mfcc_;
  Matrix<double> cmvn_stats_;
  DeltaFeaturesOptions delta_opts_;
  Matrix<BaseFloat> *features_;
};


class LaughterDetectorClassifier 
{
 public:
  LaughterDetectorClassifier(int32 threadIdx, int32 num_threads, BaseFloat threshold, Vector<BaseFloat> *laughter_prob,
                              std::vector<Vector<BaseFloat>*> *laughter_labels):
                            done_(false), threadIdx_(threadIdx), num_threads_(num_threads), threshold_(threshold),
                            laughter_prob_(laughter_prob), laughter_labels_(laughter_labels)
                            { }

  void operator() () 
  {
    // Calculate where to place the newly computed features
    MatrixIndexT length = laughter_prob_->Dim()/num_threads_;
    MatrixIndexT origin = (threadIdx_)*length;
    
    // Check if the last thread is going out if bounds, but it shouldn't
    if (threadIdx_ == num_threads_ - 1)
    {
      length = (origin + length) != laughter_prob_->Dim() ? laughter_prob_->Dim() - origin: length;
    }

    (*laughter_labels_)[threadIdx_] = new Vector<BaseFloat>(length);
    classification(laughter_prob_->Range(origin, length), *((*laughter_labels_)[threadIdx_]), threshold_);

    done_ = true;
  }

  ~LaughterDetectorClassifier()
  {
    KALDI_ASSERT(done_);
  }

 private:
  bool done_;
  int32 threadIdx_;
  int32 num_threads_;
  BaseFloat threshold_;
  Vector<BaseFloat> *laughter_prob_;
  std::vector<Vector<BaseFloat>*> *laughter_labels_;
};

/** @brief Peforms laughter detection. */
int main(int argc, char *argv[]) {

  try {
    const char *usage =
      "Performs laughter detection. \n"
      "Usage:  laughter-detector [options] <wav_file> <model_bin>\n"
      "e.g.:\n"
      " laughter-detector [options...] testFile.wav  final.nnet\n";

    ParseOptions po(usage);
    MfccOptions mfcc_ops;
    HangoverOptions hangover_ops;
    
    // Register the hangover options
    hangover_ops.Register(&po);

    // Register the mfcc options
    mfcc_ops.Register(&po);

    BaseFloat threshold = 0.9;
    po.Register("threshold", &threshold, "The threshold to apply for classifying laughter (default 0.9)");

    std::string cmvn_stats_file = "";
    po.Register("cmvn-stats", &cmvn_stats_file, "The CMVN global statistics files");

    std::string feature_transform_file = "";
    po.Register("feature-transform", &feature_transform_file, "The feature transform files");

    int32 num_threads = 1;
    po.Register("num-threads", &num_threads, "The number of threads to use (default 1)");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) 
    {
      KALDI_LOG << "Incorrect number of arguments " << po.NumArgs();
      po.PrintUsage();
      exit(1);
    }

    std::string wav_fname = po.GetArg(1);
    std::string model_fname = po.GetArg(2);

    std::string model_path = model_fname.substr(0, model_fname.find_last_of("/"));

    // File names for various results
    size_t wav_ext_point = wav_fname.find_last_of(".");
    size_t wav_filename_start_point = wav_fname.find_last_of("/") + 1;
    std::string laughter_prob_fname = "ark,t:" + wav_fname.substr(0, wav_ext_point) + ".net.out";
    std::string laughter_lables_fname = "ark,t:" + wav_fname.substr(0, wav_ext_point) + ".net.labels";
    std::string laughter_lables_hangover_fname = "ark,t:" + wav_fname.substr(0, wav_ext_point) + ".net.labels.hangover";
    std::string laughter_segments_fname = wav_fname.substr(0, wav_ext_point) + ".net.segments";
    std::string utt = wav_fname.substr(wav_filename_start_point, wav_ext_point);
    
    // File writers
    BaseFloatVectorWriter laughter_prob_writer(laughter_prob_fname);
    BaseFloatVectorWriter laughter_labels_writer(laughter_lables_fname);
    BaseFloatVectorWriter laughter_labels_hangover_writer(laughter_lables_hangover_fname);

    // Check if it is a .wav file
    if(wav_fname.substr(wav_ext_point) != ".wav")
    {
        KALDI_ERR << wav_fname << " is not a .wav file, it is: " << wav_fname.substr(wav_ext_point);
    }

    // if cmvn-stats is empty assume it is in the same location as the nnet.model
    if (cmvn_stats_file == "") 
    {
        cmvn_stats_file =  model_path + "/training_cmvn_stats";
        KALDI_LOG << "cmvn-stats option not specified, assumed to be at: " + cmvn_stats_file;
    }

    // if feature-transform" is empty assume it is in the same location as the nnet.model
    if (feature_transform_file == "") 
    {
        feature_transform_file = model_path + "/final.feature_transform";
        KALDI_LOG << "feature-transform option not specified, assumed to be at: " + feature_transform_file;
    }

    // Error Check wav file ===============================================

    // Check that the correct sample rate is being used.
    std::ifstream isinfo(wav_fname, std::ios_base::binary);
    WaveInfo winfo;
    winfo.Read(isinfo);
    isinfo.close();

    if(winfo.SampFreq() != mfcc_ops.frame_opts.samp_freq)
    {
      KALDI_ERR << wav_fname << " needs to be at a sample rate of " << mfcc_ops.frame_opts.samp_freq  << "Hz"
        << " not " << winfo.SampFreq() << "Hz";
    }

    std::ifstream iswav(wav_fname, std::ios_base::binary);
    WaveData* wave = new WaveData;
    wave->Read(iswav);
    KALDI_ASSERT(wave->Data().NumRows() == 1);
    SubVector<BaseFloat> waveform(wave->Data(), 0);
    iswav.close();

    // It is expected that laughter of very short duration (~100ms) would not be detectable
    // thus the minimum file duration is set to ~100ms
    if(waveform.Dim() < (MatrixIndexT)(0.1*mfcc_ops.frame_opts.samp_freq))
    {
        KALDI_ERR << wav_fname << " is too short, minimum duration is 100ms ";
    }

    // Multi-threaded Feature Extraction =======================================

    // MFCC object
    Mfcc mfcc(mfcc_ops);

    // CMVN Stats
    bool binary;
    Input ki(cmvn_stats_file, &binary);
    Matrix<double> cmvn_stats;
    cmvn_stats.Read(ki.Stream(), binary);

    // Create the delta options
    DeltaFeaturesOptions delta_opts;

    // Caclculate & pre-allocate the features matrix
    int32 rows_out = NumFrames(waveform.Dim(), mfcc_ops.frame_opts);
    int32 cols_out = mfcc.Dim();
    Matrix<BaseFloat>* features = new Matrix<BaseFloat>(rows_out, cols_out*(delta_opts.order + 1));
    BaseFloat max_duration_s = waveform.Dim()/mfcc_ops.frame_opts.samp_freq;
    
    g_num_threads = num_threads;
    LaughterDetectorFrontend c(&waveform, mfcc, cmvn_stats, delta_opts, features);
    RunMultiThreaded(c);
    delete wave;

    // Neural Network Processing ==============================================
    
    // Select the GPU
    CuDevice::Instantiate().SelectGpuId("yes");
    
    // Nnet transf
    Nnet* nnet_transf = new Nnet;
    nnet_transf->Read(feature_transform_file);

    // Read in the model
    Nnet* nnet = new Nnet;
    nnet->Read(model_fname);
    
    // Disable dropout,
    nnet_transf->SetDropoutRate(0.0);
    nnet->SetDropoutRate(0.0);

    CuMatrix<BaseFloat>* features_gpu = new CuMatrix<BaseFloat>;
    CuMatrix<BaseFloat>* nnet_out = new CuMatrix<BaseFloat>;
    CuMatrix<BaseFloat>* feats_transf = new CuMatrix<BaseFloat>;

    // Push the featurtes to the GPU,
    *features_gpu = *features;

    // Delete features from system memory
    delete features;

    // Feature transform,
    nnet_transf->Feedforward(*features_gpu, feats_transf);
    if (!KALDI_ISFINITE(feats_transf->Sum())) {  // check there's no nan/inf,
      KALDI_ERR << "NaN or inf found in transformed-features";
    }
    // net transdorm no longer needed
    delete nnet_transf;
    delete features_gpu;

    // Apply the forwardpass
    nnet->Feedforward(*feats_transf, nnet_out);
    if (!KALDI_ISFINITE(nnet_out->Sum())) {  // check there's no nan/inf,
      KALDI_ERR << "NaN or inf found in nn-output";
    }
    // Model no longer needed
    delete nnet;
    delete feats_transf;

    // Download the laughter probability from the GPU,
    nnet_out->Transpose();
    Vector<BaseFloat> laughter_prob = Vector<BaseFloat>(nnet_out->Row(1));
    delete nnet_out;

    int32 total_labels = laughter_prob.Dim();
    laughter_prob_writer.Write(utt, laughter_prob);

    // Classification  ==============================================
    TaskSequencerConfig config;
    config.num_threads = num_threads;
    config.num_threads_total = num_threads;

    // Create vectors for each thread
    std::vector<Vector<BaseFloat>*> laughter_labels(config.num_threads);

    TaskSequencer<LaughterDetectorClassifier> sequencer(config);
    for (int32 i = 0; i < config.num_threads; i++) 
    {
      sequencer.Run(new LaughterDetectorClassifier(i, config.num_threads, threshold,
                     &laughter_prob, &laughter_labels));
    }

    // Must have all segments before we continue
    sequencer.Wait();

    // Create a single merged laughter labels for outputting to file
    // and delete the old laughter labels
    MatrixIndexT offset = 0;
    Vector<BaseFloat>* laughter_labels_merged = new Vector<BaseFloat>(total_labels);
    for(std::vector<Vector<BaseFloat>*>::iterator threadIter = laughter_labels.begin(); threadIter != laughter_labels.end(); ++threadIter)
    {
      for(MatrixIndexT j = 0; j < (*threadIter)->Dim(); j++)
      {
        (*laughter_labels_merged)(j + offset) = (**threadIter)(j);
      }
      offset += (*threadIter)->Dim();
      delete *threadIter;
    }
    laughter_labels.clear();
    laughter_labels_writer.Write(utt, *laughter_labels_merged);


    // Segementation ==============================================
    LaughterSegment laughter_segments;
    Vector<BaseFloat>* laughter_labels_hangover = new Vector<BaseFloat>(*laughter_labels_merged);

    applyHangover(*laughter_labels_merged, *laughter_labels_hangover, laughter_segments,
                  hangover_ops, mfcc_ops.frame_opts.frame_shift_ms);
    laughter_labels_hangover_writer.Write(utt, *laughter_labels_hangover);
    
    delete laughter_labels_hangover;
    delete laughter_labels_merged;

    // Write segments with confidence =========================================
    std::ofstream laughter_segments_file(laughter_segments_fname);
    BaseFloat confidence = 0.0;
    MatrixIndexT origin = 0;
    MatrixIndexT length = 0;
    int32 decimalPrecision = 2;
    int32 timeWidth = std::to_string((int32)max_duration_s).length() + decimalPrecision + 1;

    if (laughter_segments_file.is_open())
    {
        for(LaughterSegment::iterator segmentIter = laughter_segments.begin(); segmentIter != laughter_segments.end(); ++segmentIter) 
        {
          KALDI_ASSERT(segmentIter->second >= segmentIter->first);

          origin = segmentIter->first;
          length = segmentIter->second - origin + 1;
          // Condifence is the average probability of laughter frames
          confidence = (laughter_prob.Range(origin, length)).Sum() / (length + 1);
          
          laughter_segments_file << std::setfill('0') << std::setw(timeWidth) << std::fixed << std::setprecision(decimalPrecision) << (BaseFloat)(segmentIter->first) * mfcc_ops.frame_opts.frame_shift_ms/1000 << " ";
          laughter_segments_file << std::setfill('0') << std::setw(timeWidth) << std::fixed << std::setprecision(decimalPrecision) << (BaseFloat)(segmentIter->second) * mfcc_ops.frame_opts.frame_shift_ms/1000 << " ";
          laughter_segments_file << std::fixed <<  std::setprecision(decimalPrecision) << confidence << std::endl;
        }

      laughter_segments_file.close();
    }
    else
    {
      KALDI_ERR << "Unable to open file " + laughter_segments_fname;
    }

  }
  catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}