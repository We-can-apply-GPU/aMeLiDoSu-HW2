#!/usr/bin/env python
# -*- coding: utf-8 -*-

from userCode import util

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
    print("Reading data...")
    recs = [line.strip('\n') for line in open('validRecord')]
    records=[]
    for rec in recs:
        rec = rec.rstrip().split(',')
        records += [rec]
        #print(rec)
    labs = [line.strip('\n') for line in open('data/label/validation.lab')]
    labels=[]
    for lab in labs:
        lab = lab.rstrip().split(',')
        labels += [util.char2index[lab[1]]]
      
    curPos = 0
    
    fout = open("validation.ans",'w')
    for seqName,number in records:
        print("{0},".format(seqName),file=fout,end='')
        phoneSeq = []
        for i in range(int(number)):
            phoneSeq += [labels[curPos]]
            curPos +=1
        printtrim(phoneSeq,fout)
