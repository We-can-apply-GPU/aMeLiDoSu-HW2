import json
import numpy as np
import sys

def numTo48CharVec(y):
    ans=[]
    for i in y:
        ans.append(numTo48Char(i))
    return ans

def numTo48Char(num):
    a = 'Z'
    print(num)
    chrmap = open('data/48_idx_chr.map','r')
    for line in chrmap:
        s = line.split()
        for line in s:
            if int(s[1]) == num:
                a = s[2]
                break
    return a
def predict(modelFile,inFile,outFile):

    print("Loading...\n")
    with open("model/" + modelFile,'r') as w:
        weight = json.loads(w.readline())
    datum = read_examples(inFile)

    print("Predicting...\n")
    with open("output/" + outFile,'w') as fout:
        fout.write("id,phone_sequence\n")
        for data in datum:
            y = numTo48CharVec(classify(data[1],weight))
            fout.write("{0},{1}\n".format(data[0],y))
            #fout.write("{0},{1}\n".format(data[0],ans(y)))

def read_examples(filename):
    ark = open('data/fbank/' + filename + '.ark','r')
    datum = []
    curPos = 0
    seqDic = {}

    for line in ark:
        s = line.rstrip().split(' ')
        #for line in lab:
            #l = line.rstrip().split(',')
            #if s[0] == l[0]:        #map label to train data
        for i in range(1,len(s)):
            s[i] = float(s[i])
            seqs = s[0].rstrip().split('_')
            s[0] = seqs[0] + seqs[1]
            datum.append(s)
            curPos += 1
            break
    #until now , datum is the list of [ID  FBANKfeature]
    for i in range(len(datum)):
        if(datum[i][0] not in seqDic):
            seqDic[datum[i][0]] = ([],[])
        seqDic[datum[i][0]][0].append([float(datum[i][k]) for k in range(1,len(datum[i]))])
    ans = []
    for key in seqDic:
        ans.append((key,np.array(seqDic[key][0])))
    #ans is the list of [ID FBANKfeatures]
    return ans



def classify(x,w):
    ql = list(w)
    print(len(x[0]))
    obs = np.array(ql[:69*48]).reshape((48, 69))
    trans = np.array(ql[69*48:]).reshape((48, 48))

    LEN = len(x)
    _xx = np.array(x)
    xx = _xx.reshape((LEN, 69))
    xxt = np.dot(xx, obs.T)

    y = []
    lgprob = np.zeros((48,1))
    lst = []

    for i in range(LEN):
        p = lgprob + trans + xxt[i,:]
        newlst = np.argmax(p, axis=0)
        lst.append(newlst)
        lgprob = np.max(p, axis=0).reshape((48,1))

    now = np.argmax(lgprob)
    y.append(now)
    for i in range(LEN-1, 0, -1):
        now = lst[i][now]
        y.append(now)

    y = y[::-1]
    return y
def ans(y):
    

if __name__ == '__main__':
    predict(sys.argv[1],sys.argv[2],sys.argv[3]) 
