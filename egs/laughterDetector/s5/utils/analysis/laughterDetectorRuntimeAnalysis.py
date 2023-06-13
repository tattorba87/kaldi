
import os
import math, random, sys, re
import operator
import csv
from shutil import copyfile
from subprocess import call

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt





# Main ========================================================================
def main(opts, arg):
    print("Main program started...")

    # Set figure width to 12 and height to 9
    fig_size = [18, 14]
    plt.rcParams["figure.figsize"] = fig_size


    # Loop over wav files in the directory
    for file in os.listdir(arg["wavfileDir"]):
        if file.endswith(".wav"):
            waveFile = os.path.join(arg["wavfileDir"], file)
            wavFileNameNoExt, _ = file.split(".")
            figFile = os.path.join(arg["wavfileDir"], wavFileNameNoExt + ".png")

            doesNetOutExist = os.path.isfile(os.path.join(arg["wavfileDir"], wavFileNameNoExt + ".net.out"))
            doesFigFileExist = os.path.isfile(figFile)

            # If the netout and fig exist just skip
            if doesNetOutExist and doesFigFileExist and (not opts.reset):
                continue

            print("Processing " + waveFile + "...")
        else:
            continue


        # Extract Raw Audio from Wav File
        sr, signal = wavfile.read(waveFile)
        maxSignal = np.iinfo(signal.dtype).max

        signal = signal.astype(float)

        # Normalise the signal
        signalNorm = np.divide(signal, maxSignal)

        # Invoke the laughter detector (if net output doesn't already exist)
        if not doesNetOutExist or opts.reset:
            command = [arg["ldtool"], \
                        "--threshold=" + str(opts.threshold), \
                        "--min_dur=" + str(opts.minDuration), \
                        "--sample_frequency=" + str(opts.sampleFrequency), \
                        "--use_energy=" + str(opts.useEnergy), \
                        "--min_bridge=" + str(opts.minBridge), \
                        "--num_threads=" + str(opts.numThreads), \
                        waveFile, arg["net"]]
            call(command)

        # Plot the results and save (if plot doesn't already exist)
        if not doesFigFileExist or opts.reset:
            # Read in the network output
            out_fname = os.path.splitext(waveFile)[0] + ".net.out"
            if not os.path.isfile(out_fname):
                print("WARNING: " + out_fname + " was not produced by the laughter detector. Skipped.")
                continue

            netOut = open(out_fname, "r")
            values = netOut.read()
            values = values.replace("\n", "")
            values = values.split()
            netOutFrames = values[2:-1]
            laughterProbability = np.array(netOutFrames).astype(np.float)

            # Read in the network labels
            out_fname = os.path.splitext(waveFile)[0] + ".net.labels"
            if not os.path.isfile(out_fname):
                print("WARNING: " + out_fname + " was not produced by the laughter detector. Skipped.")
                continue

            netLabelsOut = open(out_fname, "r")
            values = netLabelsOut.read()
            values = values.replace("\n", "")
            values = values.split()
            netOutFrames = values[2:-1]
            predictedLabels = np.array(netOutFrames).astype(np.int)

            # Read the network hangover labels
            out_fname = os.path.splitext(waveFile)[0] + ".net.labels.hangover"
            if not os.path.isfile(out_fname):
                print("WARNING: " + out_fname + " was not produced by the laughter detector. Skipped.")
                continue

            netLabelsOut = open(out_fname, "r")
            values = netLabelsOut.read()
            values = values.replace("\n", "")
            values = values.split()
            netOutFrames = values[2:-1]
            predictedLabelsHangover = np.array(netOutFrames).astype(np.int)

            frameShiftsamples = sr*opts.windowShift/1000
            laughterProbability = np.repeat(laughterProbability, frameShiftsamples)
            predictedLabels = np.repeat(predictedLabels, frameShiftsamples)
            predictedLabelsHangover = np.repeat(predictedLabelsHangover, frameShiftsamples)

            axes1 = plt.subplot(2, 1, 1)
            plt.plot(signalNorm)
            plt.title('Nnet Input & Output')
            plt.ylabel('Signal In')
            axes1.set_ylim([-1.1,1.1])

            axes2 = plt.subplot(2, 1, 2)
            plt.plot(laughterProbability)
            plt.plot(predictedLabels, color='r', linestyle=':', linewidth=0.5)
            plt.plot(predictedLabelsHangover, color='c', linestyle='--', linewidth=1.0)
            plt.xlabel('Samples')
            plt.ylabel('Nnet Output')
            plt.hlines(opts.threshold, 0, len(laughterProbability) - 1,  color='r', linestyles='dashed' )
            axes2.set_ylim([-0.1,1.1])

            plt.savefig(figFile)
            plt.close()

    print("Done!")


if __name__ == "__main__":

    ###
    ### Parse options
    ###
    from optparse import OptionParser
    usage = "%prog [options] <wavfileDir> <net> <laughterDetector>"
    parser = OptionParser(usage)

    parser.add_option('-s', '--windowShift_mS', type="float", dest='windowShift',
                help='Windows shift of feature extractor in mS.',
                default=10, action='store')
    parser.add_option('-t', '--threshold', type="float", dest='threshold',
                help='Threshold to apply to laughter probability.',
                default=0.9, action='store')
    parser.add_option('-d', '--minDuration', type="float", dest='minDuration',
                help='Minimum duration of the given class in mS.',
                default=240, action='store')
    parser.add_option('-b', '--minBridge', type="float", dest='minBridge',
                help='Minimum gaps of the given class to bridge in mS.',
                default=120, action='store')
    parser.add_option('-f', '--sampleFrequency', type="float", dest='sampleFrequency',
                help='The sampling frequency.',
                default=8000, action='store')        
    parser.add_option('--useEnergy', dest='useEnergy',
                help='MFCCs use energy; else C0',
                default=False, action='store_true')
    parser.add_option('-n', '--numThreads', type="int", dest='numThreads',
                help='Number of threads to run the model with.',
                default=1, action='store')
    parser.add_option('-r', '--reset', dest='reset',
                help='Do from scratch, regardless if certain files are already present',
                default=False, action='store_true')

    (opts, args) = parser.parse_args()

    if len(args) != 3 :
        parser.print_help()
        sys.exit(1)

    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    arg = {}
    arg["wavfileDir"] = args[0]
    arg["net"] = args[1]
    arg["ldtool"] = args[2]
    ### End parse options

    # Invoke main program
    main(opts, arg)