#!/usr/bin/env python
import sys
import numpy as np
from userCode import util

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
                print "123"
                fout.write("{0}_{1},".format(speaker_id, sequence_id))
                ans = util.viterbi(data, weight)
                for start in range(len(ans)):
                    if ans[start] != 36:
                        break
                for end in range(len(ans)-1, -1, -1):
                    if ans[end] != 36:
                        break
                pre = -1
                for index in range(start, end+1):
                    if ans[index] == pre:
                        continue
                    fout.write(util.index2char[ans[index]])
                    pre = ans[index]
                fout.write("\n")
    fout.close()

