import os
import math, random, sys, re
import operator
import csv
from shutil import copyfile
from subprocess import call
from itertools import chain
import numpy as np
from scipy.io import wavfile
from scipy.stats import gamma
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


# Resample & convert bit rate the files =======================================
#      SSRC : A fast and high quality sampling rate converter
#                                         written by Naoki Shibata
# Homepage : http://shibatch.sourceforge.net/
# e-mail   : shibatch@users.sourceforge.net

def convert_SR_and_bith_depth(srcDataDir, destDataDir, srcTool, sampleRate, bitDepth):
    print("Converting sample rate {} Hz and bit depth {} bits... ".format(sampleRate, bitDepth))

    if os.path.isdir(destDataDir):
        print(destDataDir + " already exists. Skipping the SRC.")
    else:
        os.mkdir(destDataDir)

        for filename in os.listdir(srcDataDir):
            if filename.endswith(".wav"):
                filenameSrc = os.path.join(srcDataDir, filename)
                filenameDest = os.path.join(destDataDir, filename)
                command = [srcTool,"--quiet" , "--rate", str(sampleRate), "--bits", str(bitDepth),  filenameSrc, filenameDest] 
                call(command)
            else:
                continue

    print("Done.")


# Get the laughter events =====================================================

def getLaughterSegments(dataDir, laughterSegmentsDir, nonLaughterSegmentsDir, masterLabelFile, newLabelFile, sampleRate, bitDepth, minLength_ms):
    print("....")
    print("Extracting laughter segments for " + dataDir + "...")

    if os.path.isdir(laughterSegmentsDir):
        print(laughterSegmentsDir + " already exists.")
    else:
        os.mkdir(laughterSegmentsDir)

    if os.path.isdir(nonLaughterSegmentsDir):
        print(nonLaughterSegmentsDir + " already exists.")
    else:
        os.mkdir(nonLaughterSegmentsDir)

    nonLaughterSegmentsSubsetDir = nonLaughterSegmentsDir + '_subset'
    if os.path.isdir(nonLaughterSegmentsSubsetDir):
        print(nonLaughterSegmentsSubsetDir + " already exists.")
    else:
        os.mkdir(nonLaughterSegmentsSubsetDir)

    # Open the new labels file
    if os.path.exists(newLabelFile):
        labelsFile = open(newLabelFile, "a")
    else:
        labelsFile = open(newLabelFile, "w+")

    # Create an utt2spk file which is the same as the label file,
    # but this will only be for the given dataset.
    utt2spkFileName = os.path.dirname(dataDir) + "/utt2spk"
    utt2spkFile = open(utt2spkFileName, "w+")

    spk2uttFileName = os.path.dirname(dataDir) + "/spk2utt"
    spk2uttFile = open(spk2uttFileName,"w+")

    # Create wav.scp file
    wavscpFileName = os.path.dirname(dataDir) + "/wav.scp"
    wavscpFile = open(wavscpFileName, "w+")

    # Build a map of all the files that contain laughter and non-laughter----------------------
    # Ignore utternaces that do not contain laughter
    laughterSegmentsMap = {}
    laughterSpeakerIDMap = {}

    with open(masterLabelFile) as mf:
        fileList = [line.split(",") for line in mf]
    
    laughterCount = 0
    startSeg = lambda i: i + 1
    endSeg = lambda i: i + 2
    for line in fileList:              #print the list items
        laughterSegments = list(chain.from_iterable((startSeg(i), endSeg(i)) for i, s in enumerate(line) if 'laughter' in s ))
        if laughterSegments:
            laughterSegments = map(lambda x: int(sampleRate*float(line[x])) , laughterSegments)
            laughterSegments = zip(laughterSegments[0::2], laughterSegments[1::2])
            laughterCount = laughterCount + len(laughterSegments)
            laughterSegmentsMap[line[0]] = laughterSegments
            laughterSpeakerIDMap[line[0]] = line[1]


    # Save all of the laughter segments----------------------------------------
    print("Total number of laughter segments in the labels file = " + str(laughterCount))
    laughterLengthData = np.zeros(laughterCount)
    laughterKept = 0
    laughterDiscarded = 0

    # Build the complementary non-laughter files
    nonLaughterSegmentsLengthMap = {}
    
    for uttID, laughterSegments in laughterSegmentsMap.items():
        
        # If file is not int the dataset just skip, it is fine, it might be part of a different dataset (Train, dev or Test)
        uttFilename = os.path.join(dataDir, uttID + ".wav")

        if not os.path.isfile(uttFilename):
            continue

        sr, utterancePCM = wavfile.read(uttFilename)

        if sr != sampleRate:
            sys.exit("Sample rate mismatch!")

        nFramesUtterance = len(utterancePCM)

        # For every laughter segment save it as a new wave file, create the wav.scp file
        # create the UTT2PK
        # Name of new files. S0001_L01 or S0001_G01 
        
        nonLaughterStartFrame = 0
        laughterIdx = 1
        nonLaughterIdx = 1
        for _, laughterSegment in enumerate(laughterSegments):
            laughterFileName = uttID + "_L"  + str(laughterIdx).zfill(2)
            
            # Discard any laughter that is < minLength_ms.
            laughterLength_ms = 1000*(laughterSegment[1] - laughterSegment[0] + 1)/sampleRate
            if laughterLength_ms < minLength_ms:
                laughterDiscarded = laughterDiscarded + 1
                continue

            if laughterSegment[1] < laughterSegment[0]:
                sys.exit("laughterSegment[1] ({}) < laughterSegment[0] ({})".format(laughterSegment[1], laughterSegment[0]))

            if laughterSegment[0] >= nFramesUtterance:
                sys.exit("laughterSegment[0] ({}) >= nFramesUtterance ({})".format(laughterSegment[0], nFramesUtterance))

            if laughterSegment[0] >= nFramesUtterance:
                sys.exit("laughterSegment[0] ({}) >= nFramesUtterance ({})".format(laughterSegment[0], nFramesUtterance))

            labelsFile.write("{}\n".format(laughterFileName + ',' + laughterSpeakerIDMap[uttID]))
            utt2spkFile.write("{}\n".format(laughterFileName + ' ' + laughterSpeakerIDMap[uttID]))

            laughterLengthData[laughterKept] = laughterLength_ms
            laughterKept = laughterKept + 1
            laughterIdx = laughterIdx + 1
            wavfile.write(os.path.join(laughterSegmentsDir, laughterFileName + ".wav"), sampleRate, utterancePCM[ laughterSegment[0]:laughterSegment[1] ])
            wavscpFile.write("{}\n".format(laughterFileName + " " + os.path.abspath(os.path.join(laughterSegmentsDir, laughterFileName + ".wav"))))


            # Write non-laughter files
            if nonLaughterStartFrame < laughterSegment[0]:
                nonLaughterLength_ms = 1000*((laughterSegment[0] - 1)  - nonLaughterStartFrame + 1)/sampleRate

                if nonLaughterLength_ms > minLength_ms:
                    nonLaughterFileName = uttID + "_NL"  + str(nonLaughterIdx).zfill(2)

                    wavfile.write(os.path.join(nonLaughterSegmentsDir, nonLaughterFileName + ".wav"), sampleRate, utterancePCM[ nonLaughterStartFrame:(laughterSegment[0] - 1) ])
                    # Save lengths
                    nonLaughterSegmentsLengthMap[nonLaughterFileName] = nonLaughterLength_ms
                    # Update start of next non-laughter segement
                    nonLaughterStartFrame = laughterSegment[1] + 1
                    nonLaughterIdx = nonLaughterIdx + 1


        # Check if the file ends with non-laughter
        if nonLaughterStartFrame < nFramesUtterance:
            nonLaughterLength_ms = 1000*((nFramesUtterance - 1)  - nonLaughterStartFrame + 1)/sampleRate

            if nonLaughterLength_ms > minLength_ms:
                nonLaughterFileName = uttID + "_NL"  + str(nonLaughterIdx).zfill(2)  
                wavfile.write(os.path.join(nonLaughterSegmentsDir, nonLaughterFileName + ".wav"), sampleRate, utterancePCM[ nonLaughterStartFrame:(nFramesUtterance - 1) ])
                nonLaughterSegmentsLengthMap[nonLaughterFileName] = nonLaughterLength_ms


    print("Total number of laughter segments  discarded (< " + str(minLength_ms) + "ms) = " + str(laughterDiscarded))
    print("Total number of laughter segments (with discarding) = " + str(laughterKept))
    print("Total number of laughter segments not in " + dataDir + " = " + str(laughterCount - laughterDiscarded - laughterKept))
    print("Total number of non-laughter segments  = " + str(len(nonLaughterSegmentsLengthMap)))

    sorted_nonLaughterSegmentsLengthList = sorted(nonLaughterSegmentsLengthMap.items(), key=operator.itemgetter(1))

    # Shorten the laughterLengthData (remove the space that was not used)
    laughterLengthData= np.resize(laughterLengthData, laughterKept)

    # Get the total duration of laughter in ms
    totalLaughter_ms = sum(laughterLengthData)

    # Get the shuffled list of segements
    nonLaughterShuffledIdx = random.sample(range(1, len(nonLaughterSegmentsLengthMap) + 1), len(nonLaughterSegmentsLengthMap))

    #uttID, laughterSegments in laughterSegmentsMap.items():

    # Loop over the segments until we reach the desired length
    totalNonLaughter_ms = 0
    nonLaughterCount = 0
    while  totalNonLaughter_ms < totalLaughter_ms:
        idx = nonLaughterShuffledIdx[nonLaughterCount]
        nonLaughterFileSrc = os.path.join(nonLaughterSegmentsDir, sorted_nonLaughterSegmentsLengthList[idx][0] + ".wav")
        nonLaughterFileDst = os.path.join(nonLaughterSegmentsSubsetDir, sorted_nonLaughterSegmentsLengthList[idx][0] + ".wav")
        copyfile(nonLaughterFileSrc, nonLaughterFileDst)
        totalNonLaughter_ms = totalNonLaughter_ms + sorted_nonLaughterSegmentsLengthList[idx][1]

        uttID = sorted_nonLaughterSegmentsLengthList[idx][0].split('_')[0]
        labelsFile.write("{}\n".format(sorted_nonLaughterSegmentsLengthList[idx][0]  + ',' + laughterSpeakerIDMap[uttID]))
        utt2spkFile.write("{}\n".format(sorted_nonLaughterSegmentsLengthList[idx][0]  + ' ' + laughterSpeakerIDMap[uttID]))

        wavscpFile.write("{}\n".format(sorted_nonLaughterSegmentsLengthList[idx][0] + " " + os.path.abspath(nonLaughterFileDst)))

        nonLaughterCount = nonLaughterCount + 1  

    labelsFile.close()
    utt2spkFile.close()
    wavscpFile.close()

    # Produce a sorted version of the utt2spk file
    reader = csv.reader(open(utt2spkFileName), delimiter=" ")
    sortedlist = sorted(reader, key=operator.itemgetter(0), reverse=False)

    with open(utt2spkFileName + ".sorted" , "w") as output:
        writer = csv.writer(output, lineterminator='\n', delimiter=" ")
        writer.writerows(sortedlist)

    # Delete utt2spk and replace with the sorted version
    if os.path.exists(utt2spkFileName):
        os.remove(utt2spkFileName)
        os.rename(utt2spkFileName + ".sorted", utt2spkFileName)

    # Call the perl script which converts utt2spk to spk2utt
    utt2spk_to_spk2utt_script = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/utt2spk_to_spk2utt.pl"
    command = ["perl", utt2spk_to_spk2utt_script, utt2spkFileName] 
    call(command, stdout=spk2uttFile, shell=False)
    spk2uttFile.close()

    # Produce a sorted version of the wav.scp file
    reader = csv.reader(open(wavscpFileName), delimiter=" ")
    sortedlist = sorted(reader, key=operator.itemgetter(0), reverse=False)

    with open(wavscpFileName + ".sorted" , "w") as output:
        writer = csv.writer(output, lineterminator='\n', delimiter=" ")
        writer.writerows(sortedlist)

    # Delete wav.scp and replace with the sorted version
    if os.path.exists(wavscpFileName):
        os.remove(wavscpFileName)
        os.rename(wavscpFileName + ".sorted", wavscpFileName)
    
    print("Done.")


# Main ========================================================================
def main(opts, arg):
    print("Main program started...")

    random.seed(777)
    
    # SRC & Bitrate to 16 bits
    destTrainDataDir = arg["dataTrain"] + "_{}Hz_{}bits".format(arg["sampleRate"], arg["bitDepth"])
    #convert_SR_and_bith_depth(arg["dataTrain"], destTrainDataDir, arg["srcTool"], arg["sampleRate"], arg["bitDepth"])
    destTestDataDir = arg["dataTest"] + "_{}Hz_{}bits".format(arg["sampleRate"], arg["bitDepth"])
   # convert_SR_and_bith_depth(arg["dataTest"], destTestDataDir, arg["srcTool"], arg["sampleRate"], arg["bitDepth"])
    
    # Delete old labels file
    if os.path.exists(arg["newLabelsFile"]):
        os.remove(arg["newLabelsFile"])

    # Get Laughter segments & NL segments for training
    destTrainLaughterDataDir = arg["dataTrain"] + "_laughter_" + "{}Hz_{}bits".format(arg["sampleRate"], arg["bitDepth"])
    destTrainNonLaughterDataDir = arg["dataTrain"]  + "_nonlaughter_" + "{}Hz_{}bits".format(arg["sampleRate"], arg["bitDepth"])
    getLaughterSegments(destTrainDataDir, destTrainLaughterDataDir, destTrainNonLaughterDataDir, arg["masterLabelsFile"], arg["newLabelsFile"], arg["sampleRate"], arg["bitDepth"], 100)
    
    # Get Laughter segments & NL segments for test
    destTestLaughtertDataDir = arg["dataTest"] + "_laughter_" + "{}Hz_{}bits".format(arg["sampleRate"], arg["bitDepth"])
    destTestNonLaughterDataDir = arg["dataTest"] + "_nonlaughter_" + "{}Hz_{}bits".format(arg["sampleRate"], arg["bitDepth"])
    getLaughterSegments(destTestDataDir, destTestLaughtertDataDir, destTestNonLaughterDataDir, arg["masterLabelsFile"], arg["newLabelsFile"], arg["sampleRate"], arg["bitDepth"], 100)
    
    # Produce a sorted version of the new labels
    reader = csv.reader(open(arg["newLabelsFile"]), delimiter=",")
    sortedlist = sorted(reader, key=operator.itemgetter(0), reverse=False)

    with open(arg["newLabelsFile"] + ".sorted" , "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(sortedlist)

    # Delete labels file and replace with sorted version
    if os.path.exists(arg["newLabelsFile"]):
        os.remove(arg["newLabelsFile"])
        os.rename(arg["newLabelsFile"] + ".sorted", arg["newLabelsFile"])

    print("\nMain program terminated")


if __name__ == "__main__":

    ###
    ### Parse options
    ###
    from optparse import OptionParser
    usage = "%prog [options] <training-data> <test-data> <masterLabelsFile> <newLabelsFile> <src-tool> <sample-rate> <bit-depth> "
    parser = OptionParser(usage)

    parser.add_option('--quiet', dest='quiet',
                    help='nada',
                    default=False, action='store_false')


    (opts,args) = parser.parse_args()
    if len(args) != 7 :
        parser.print_help()
        sys.exit(1)

    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    arg = {}
    arg["dataTrain"] = args[0]
    arg["dataTest"] = args[1]
    arg["masterLabelsFile"] = args[2]
    arg["newLabelsFile"] = args[3]
    arg["srcTool"] = args[4]
    (arg["sampleRate"], arg["bitDepth"]) = map(int, args[5:])
    ### End parse options

    # Invoke main program
    main(opts, arg)
    