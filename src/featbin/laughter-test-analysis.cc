// nnetbin/laughterLabels-to-post.cc

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

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


/** @brief Peforms analysis of the laughter detector outputs. */
int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  
  try {
    const char *usage =
      "Peforms analysis of the laughter detector outputs \n"
      "Usage:  laughter-test-analysis [options] <wav_rspecifier> <results_rspecifier>\n"
      "e.g.:\n"
      " laughter-test-analysis [options...]  scp,p:data/test/wav.scp  ark:data/test/results/results.ark\n";

    ParseOptions po(usage);

    float threshold = 0.5;
    po.Register("threshold", &threshold, "The threshold to apply for classifying laughter (deault 0.5)");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) 
    {
      KALDI_LOG << "Incorrect number of arguments " << po.NumArgs();
      po.PrintUsage();
      exit(1);
    }

    std::string wav_rspecifier = po.GetArg(1);
    std::string results_rspecifier = po.GetArg(2);
    
    // Read wav.scp
    SequentialTableReader<WaveHolder> wavReader(wav_rspecifier);

    // Read the results.ark
    RandomAccessBaseFloatMatrixReader resultsReader(results_rspecifier);

    int64 falsePositives = 0;
    int64 falseNegatives = 0;
    int64 truePositives = 0;
    int64 trueNegatives = 0;
    int64 totalFrames = 0;
    int32 label = 0;
    int32 netLabel = 0;
    const std::string notLaughter = "NL";
    const std::string laughter = "L";

    for (; !wavReader.Done(); wavReader.Next()) 
    {
      std::string utt = wavReader.Key();
      //const WaveData &wave_data = wavReader.Value();

      // Is Utterance laughter or non-laughter?
      if(utt.find(notLaughter) != std::string::npos)
      {
        label = 0;
      }
      else if(utt.find(laughter) != std::string::npos)
      {
          label = 1;
      }
      else
      {
          continue;
      }

      Matrix<BaseFloat> netUttOutput = resultsReader.Value(utt);

      for(int32 row = 0; row < netUttOutput.NumRows(); row++)
      {
        float* netFrameOutput  = netUttOutput.RowData(row);

        // Does the nnet think the frame is laughter?
        if(netFrameOutput[1] >= threshold)
        {
          netLabel = 1;
        }
        else
        {
          netLabel = 0;
        }

        // Sum results
        if (netLabel == 1 && label == 1)
        {
            truePositives++;
        }
        else if (netLabel == 0 && label == 0)
        {
            trueNegatives++;
        }
        else if (netLabel == 1 && label == 0)
        {
            falsePositives++;
        }
        else if (netLabel == 0 && label == 1)
        {
            falseNegatives++;
        }
        else
        {
          KALDI_ASSERT(0);
        }

        totalFrames++;
      }
    }

    float precision = (float)truePositives/(truePositives + falsePositives);
    float recall = (float)truePositives/(truePositives + falseNegatives);

    KALDI_LOG  << "";
    KALDI_LOG  << "========================================";
    KALDI_LOG  << "Laughter Detector Results";
    KALDI_LOG  << "========================================";
    KALDI_LOG  << "";
    KALDI_LOG  << "Total frames tested " << totalFrames;
    KALDI_LOG  << "";
    KALDI_LOG  << "True positives = " << 100*(float)truePositives/totalFrames << "%";
    KALDI_LOG  << "True negatives = " << 100*(float)trueNegatives/totalFrames << "%";
    KALDI_LOG  << "False positives = " << 100*(float)falsePositives/totalFrames << "%";
    KALDI_LOG  << "False negatives = " << 100*(float)falseNegatives/totalFrames << "%";
    KALDI_LOG  << "";
    KALDI_LOG  << "Precision = " << 100*precision << "%";
    KALDI_LOG  << "Recall = " << 100*recall << "%";
    KALDI_LOG  << "F1 score = " << 100*(float)2*precision*recall/(precision + recall) << "%";
    KALDI_LOG  << "========================================";

  }
  catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}