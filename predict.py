#!/usr/bin/env python
from __future__ import print_function
import sys
import numpy as np
from userCode import util

PHONES = 48
FBANKS = 69

def printtrim(ans, fout):
    for start in range(len(ans)):
        if ans[start] != 37:
            break
    for end in range(len(ans)-1, -1, -1):
        if ans[end] != 37:
            break
    pre = -1
    for index in range(start, end+1):
        if ans[index] == pre:
            continue
        print(util.index2ans[ans[index]], file=fout, end='')
        pre = ans[index]
    print(file=fout)

if __name__ == "__main__":
    print("Reading data")
    fbank = util.read_fbank("test")
    weight = np.array(util.read_weight(sys.argv[1]))
    print("Processing")
    fout = open(sys.argv[1]+".ans", "w")
    fout.write("id,phone_sequence")
    fout.write("\n")
    ftmp = []
    print("{0}_{1},".format(fbank[0][0][0], fbank[0][0][1]), file=fout, end='')
    for f in fbank:
        if f[0][2] == 1 and len(ftmp) != 0:
            ans = util.viterbi(np.array(ftmp), weight)
            printtrim(ans, fout)
            ftmp = []
            print("{0}_{1},".format(f[0][0], f[0][1]), file=fout, end='')
        ftmp += [f[1:]]
    if len(ftmp) != 0:
        ans = util.viterbi(np.array(ftmp), weight)
        printtrim(ans, fout)
    fout.close()
