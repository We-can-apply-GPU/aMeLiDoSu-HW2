#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: feature_train.py
Description: feature extraction
"""
import numpy as np

dic = {}
feature = np.zeros(5616)

def infile():
    ark = open('../data/fbank/trainToy.ark','r')
    lab = open('../data/label/trainToy.lab','r')
    for line in ark:
        s = line.rstrip().split(' ')
        for line in lab:
            l = line.rstrip().split(',')
            if s[0] == l[0]:        #map label to train data
                for i in range(1,len(s)):
                    s[i] = float(s[i])
                s.append(l[1])
                dic[len(dic)]=s 
                #print(dic[len(dic)-1])
                break               
    #dic=[Instance ID + feature + label]

def charto48(c):
    chrmap = open('../data/48_idx_chr.map','r')
    for line in chrmap:
        s = line.split()
        for line in s:
            if s[0] == c:
                num = int(s[1])
                break
    return num

def extract(start, end):
    for i in range (start, end):
        num = charto48(dic[i][-1])
        if i != end-1:
            num2 = charto48(dic[i+1][-1])
            feature[48*69+48*num+num2] += 1
        for j in range (0, 69):
            feature[num*69+j] += dic[i][1+j]
    outfile(dic[start][0])          #output specific utterance feature

def outfile(ID):
    ID=ID[0:-2]
    f=open('feature.CSV','a')
    f.write(ID + ' ')
    for i in range (0, len(feature)):
        f.write(str(feature[i]) + ' ')
    f.write('\n')

if __name__=="__main__":
    infile()
    nowID=dic[0][0];                #dic=[Instance ID + feature + label]
    nowID=nowID[0:-2];              #cut out _0
    startnum=0;                     #specific utterance start from
    for i in range (0,len(dic)):
        if nowID not in dic[i][0]:
            extract(startnum,i)
            startnum=i
            nowID=dic[i][0];
            nowID=nowID[0:-2]
            feature = np.zeros(5616)  #initial
        if i == len(dic)-1:
            extract(startnum,len(dic))

