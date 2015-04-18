#!/usr/bin/env python
import sys
import numpy as np
import util

PHONES = 48
FBANKS = 69

if __name__ == "__main__":
    print("Reading data")
    fbank = util.read_fbank("test")
    weight = util.read_weight(sys.argv[1])
    print("Processing")
    fout = open("result", "w")
    fout.write("id,phone_sequence")
    fout.write("\n")
    for data in fbank:
        for (i, (speaker_id, v)) in enumerate(fbank.iteritems()):
            for (sequence_id, data) in v.iteritems():

                print("{0}_{1},".format(speaker_id, sequence_id))
                ans = viterbi(data, weight)
                for index in ans:
                    print(util.index2char[index])
                print("\n")

