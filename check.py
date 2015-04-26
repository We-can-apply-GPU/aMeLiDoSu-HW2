#!/usr/bin/env python
from __future__ import print_function
import sys
from userCode import util
import numpy as np

fbank = util.read_fbank("train")
label = util.read_label("train")
weight = np.array(util.read_weight(sys.argv[1]))

ftmp = []
ltmp= []
cnt = 0
lenseq = 0

for f, l in zip(fbank, label):
    if f[0][2] == 1 and len(ftmp) != 0:
        ans = util.viterbi(np.array(ftmp), weight)
        lenseq += len(ans)
        for aa, ll in zip(ans, ltmp):
            if aa != ll:
                cnt += 1
        ftmp = []
        ltmp = []
    ftmp += [f[1:]]
    ltmp += [l[1]]

if len(ftmp) != 0:
    ans = util.viterbi(np.array(ftmp), weight)
    lenseq += len(ans)
    for aa, ll in zip(ans, ltmp):
        if aa != ll:
            cnt += 1

print(100.0 * cnt / lenseq)
