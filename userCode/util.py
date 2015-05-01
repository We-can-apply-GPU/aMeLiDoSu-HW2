import numpy as np
index2char = {}
char2index = {}
index2ans = {}
char2ans={}
a = 0
chrmap = open('data/48_idx_chr.map','r')
for line in chrmap:
    line = line.split()
    index2char[int(line[1])] = line[0]
    char2index[line[0]] = int(line[1])
    index2ans[int(line[1])] = line[2]
    char2ans[line[0]] = line[2]
PHONES = 48
FBANKS = 69

def read_fbank(filename):
    fin = open("data/fbank/" + filename +".ark", "r")
    lines = []
    for line in fin:
        line = line.rstrip().split(' ')
        line[0] = line[0].split('_')
        line[0][2] = int(line[0][2])
        line[1:] = [float(ll) for ll in line[1:]]
        lines += [line]
    return lines

def read_label(filename):
    fin = open("data/label/" + filename + ".lab", "r")
    lines = []
    for line in fin:
        line = line.rstrip().split(',')
        line[0] = line[0].split('_')
        line[0][2] = int(line[0][2])
        line[1] = char2index[line[1]]
        lines += [line]
    return lines

def read_weight(filename):
    import json
    f = open("model/" + filename, "r")
    return json.loads(f.readline())

def viterbi(x, w, y = [],hw1Rate = 0,indexCnt=0,hw1MatList=[]):
    w = list(w)
    lenx = len(x)
    x = x.reshape((lenx, FBANKS))

    observation = np.array(w[:PHONES*FBANKS]).reshape((PHONES, FBANKS))

    trans = np.array(w[PHONES*FBANKS:]).reshape((PHONES, PHONES))
    xobs = np.dot(x, observation.T)
    print(xobs)
    if(len(hw1MatList) != 0): #use  hw1
        xobs = xobs + (1 * hw1MatList[indexCnt])

    prob_pre = np.zeros((PHONES, 1))
    trace = []
    for i in range(lenx):
        prob_now = prob_pre + trans + xobs[i, :]
        if len(y) > 0:
            prob_now[:,y[i]] -= 1
        argmax = np.argmax(prob_now, axis = 0)
        prob_pre = np.max(prob_now, axis = 0).reshape((PHONES, 1))
        trace.append(argmax)
    now = np.argmax(prob_pre)
    ans = []
    ans.append(now)
    for i in range(lenx-1, 0, -1):
        now = trace[i][now]
        ans.append(now)
    return np.array(ans[::-1])
def hw1data(f ='data/HW1data'):
    d = open(f)
    now=['people','talk']
    ls = []
    m = []
    for i in d:
        s = i.rstrip().split(',[')
        #print
        #if s[1].__len__()==0:
            #print(s)
        n = s[0].split('_')
        s = (s[1][:-1]).split(', ')
        if now[0]!='people':
            if now[0] != n[0] or now[1] != n[1]:
                ls.append(np.array(m))
                m = []
        m.append([ float(i) for i in s])
        now[0] = n[0]
        now[1] = n[1]
    ls.append(np.array(m))
    return ls
