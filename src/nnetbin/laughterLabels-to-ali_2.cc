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
#include "hmm/posterior.h"
#include "feat/feature-mfcc.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

/** @brief Converts the laughter labels into an alignment archive file, this can be than converted to a posterior format
 *  which is the generic format of NN training targets in 'nnet1'. */


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
      "Converts the laughter labels into alignments, this can be than converted to a posterior format \n"
      " which is the generic format of NN training targets in Karel's nnet1 tools.\n"
      "Usage:  laughterLabels-to-ali_2 [options]  <utt2num_frames> <alignment-wspecifier>\n"
      "e.g.:\n"
      " laughterLabels-to-ali_2  [options...]  utt2num_frames  ali.ark\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 2)
    {
      KALDI_LOG << "Incorrect number of arguments " << po.NumArgs();
      po.PrintUsage();
      exit(1);
    }

    std::string utt2num_frames_file   = po.GetArg(1);
    std::string ali_wspecifier = po.GetArg(2);

    // Open the utt2num_frames file
    std::ifstream inUttFrameFile(utt2num_frames_file, std::ios::in);

    if(!inUttFrameFile)
    {
        KALDI_ERR << utt2num_frames_file << " could not be opened" << std::endl;
    }

    Int32VectorWriter ali_writer(ali_wspecifier);
    std::string line;
    std::string frameUttId;
    int32 numFrames = 0;
    int32 label = 0;
    const std::string notLaughter = "NL";
    const std::string laughter = "L";

    while(!inUttFrameFile.eof())
    {
        // Get the next frame uttID that we need
        getline(inUttFrameFile, line);
        std::istringstream ssLine(line);
        getline(ssLine, frameUttId, ' ');
        ssLine >>  numFrames;

        if(frameUttId.find(notLaughter) != std::string::npos)
        {
            label = 0;
        }
        else if(frameUttId.find(laughter) != std::string::npos)
        {
            label = 1;
        }
        else
        {
            continue;
        }

        // Create the vector of labels (alignments)
        std::vector<int32> target(numFrames, label);

        // Write the alignment to the ark file
        ali_writer.Write(frameUttId, target);
    } 
  }
  catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
