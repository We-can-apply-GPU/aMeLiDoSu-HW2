#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: feature_hw.py
Description: feature exaction
"""
import numpy as np

dic = {}
feature = np.zeros(5616)

def infile():
    ark = open('../data/fbank/HW2_a.ark','r')
    lab = open('../data/label/HW2_a.lab','r')
    for line in ark:
        s = line.rstrip().split(' ')
        for line in lab:
            l = line.rstrip().split(',')
            if s[0] == l[0]:       #map label to train data
                for i in range(1,len(s)):
                    s[i] = float(s[i])
                s.append(l[1])
                dic[len(dic)]=s 
                #print(dic[len(dic)-1])   #dic=[Instance ID + feature + label]
                break

def charto48(c):
    chrmap = open('../data/48_idx_chr.map','r')
    for line in chrmap:
        s = line.split()
        for line in s:
            if s[0] == c:
                num = int(s[1])
                break
    return num

def exact():
    for i in range (0, len(dic)):
        num = charto48(dic[i][-1])
        if i != len(dic)-1:
            num2 = charto48(dic[i+1][-1])
            feature[48*69+48*num+num2] += 1
        for j in range (0, 69):
            feature[num*69+j] += dic[i][1+j]

def outfile(ID):
    ID=ID[0:-1]
    f=open('HW2_a.CSV','w')
    f.write('id,feature\n')
    for i in range (0, len(feature)):
        f.write(ID + str(i) + ',' + str(feature[i]) + '\n')

if __name__=="__main__":
    infile()
    exact()
    outfile(dic[0][0])
