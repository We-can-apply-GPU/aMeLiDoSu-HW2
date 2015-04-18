#!/usr/bin/env python
#<<<<<<< HEAD
 #-*- coding: utf-8 -*-
#import json
#import numpy as np
#import sys

#def numTo48CharVec(y):
    #ans=[]
    #for i in y:
        #ans.append(numTo48Char(i))
    #return ans

#def numTo48Char(num):
    #a = 'Z'
    #chrmap = open('data/48_idx_chr.map','r')
    #for line in chrmap:
        #s = line.split()
        #for line in s:
            #if int(s[1]) == num:
                #a = s[2]
                #break
    #return a
#def predict(modelFile,inFile,outFile):

    #print("Loading...")
    #with open("model/" + modelFile,'r') as w:
        #weight = json.loads(w.readline())
    #datum = read_examples(inFile)

    #print("Predicting...")
    #with open("output/" + outFile,'w') as fout:
        #fout.write("id,phone_sequence\n")
        #for data in datum:
            #y = numTo48CharVec(classify(data[1],weight))
            #fout.write("{0},{1}\n".format(data[0],ans(y)))

#def read_examples(filename):
    #ark = open('data/fbank/' + filename + '.ark','r')
    #datum = []
    #curPos = 0
    #seqDic = {}

    #for line in ark:
        #s = line.rstrip().split(' ')
        #for line in lab:
            #l = line.rstrip().split(',')
            #if s[0] == l[0]:        #map label to train data
        #for i in range(1,len(s)):
            #s[i] = float(s[i])
            #seqs = s[0].rstrip().split('_')
            #s[0] = seqs[0] + '_' + seqs[1]
            #datum.append(s)
            #curPos += 1
            #break
    #until now , datum is the list of [ID  FBANKfeature]
    #for i in range(len(datum)):
        #if(datum[i][0] not in seqDic):
            #seqDic[datum[i][0]] = ([],[])
        #seqDic[datum[i][0]][0].append([float(datum[i][k]) for k in range(1,len(datum[i]))])
    #ans = []
    #for key in seqDic:
        #ans.append((key,np.array(seqDic[key][0])))
    #ans is the list of [ID FBANKfeatures]
    #return ans



#def classify(x,w):
    #ql = list(w)
    #print(len(x[0]))
    #obs = np.array(ql[:69*48]).reshape((48, 69))
    #trans = np.array(ql[69*48:]).reshape((48, 48))

    #LEN = len(x)
    #_xx = np.array(x)
    #xx = _xx.reshape((LEN, 69))
    #xxt = np.dot(xx, obs.T)

    #y = []
    #lgprob = np.zeros((48,1))
    #lst = []

    #for i in range(LEN):
        #p = lgprob + trans + xxt[i,:]
        #newlst = np.argmax(p, axis=0)
        #lst.append(newlst)
        #lgprob = np.max(p, axis=0).reshape((48,1))

    #now = np.argmax(lgprob)
    #y.append(now)
    #for i in range(LEN-1, 0, -1):
        #now = lst[i][now]
        #y.append(now)

    #y = y[::-1]
    #return y
#def ans(y):
    #answer = ""
    #start = False
    #pre = 'K'
    #for char in y:
        #now = char
        #if( not (start)):
            #if now == 'K': sil
                #continue
            #else:
                #start = True
        #if(now != pre):
            #answer += now
            #pre = now
    #return answer

#if __name__ == '__main__':
    #predict(sys.argv[1],sys.argv[2],sys.argv[3]) 
#=======
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

#>>>>>>> structure
