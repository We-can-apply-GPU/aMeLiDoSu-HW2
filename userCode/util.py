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

def viterbi(x, w, y = [],hw1Rate = 0,hw1Mat=np.zeros(0)):
    w = list(w)
    lenx = len(x)
    x = x.reshape((lenx, FBANKS))

    observation = np.array(w[:PHONES*FBANKS]).reshape((PHONES, FBANKS))
    trans = np.array(w[PHONES*FBANKS:]).reshape((PHONES, PHONES))
    xobs = np.dot(x, observation.T)

    #print(xobs.shape == hw1Mat.shape)
    if(hw1Rate !=0): #use  hw1
        xobs = np.log(hw1Mat) * hw1Rate + xobs
        #xobs = hw1Mat * hw1Rate

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

def hw1Preprocess(ls):
    ll = ls
    if(len(ls) != 48):
        print("ERROR")
    else:
        ll[29] = ls[30]
        ll[30] = ls[29]
        ll[35] = ls[37]
        ll[36] = ls[35]
        ll[37] = ls[36]
        ll[38] = ls[39]
        ll[39] = ls[38]
        ll[42] = ls[43]
        ll[43] = ls[42]
        ll[46] = ls[47]
        ll[47] = ls[46]
        #ls[29],ls[30] = ls[30],ls[29]
        #ls[35],ls[36] = ls[36],ls[35]
        #ls[36],ls[37] = ls[37],ls[36]
        #ls[38],ls[39] = ls[39],ls[38]
        #ls[42],ls[43] = ls[43],ls[42]
        #ls[46],ls[47] = ls[47],ls[46]
    return ll

def hw1data(f ='data/HW1data'):
    d = open(f)
    now=['people','talk']
    ls = []
    m = []
    for i in d:
        s = i.rstrip().split(',[')
        n = s[0].split('_')
        s = (s[1][:-1]).split(', ')

        s = hw1Preprocess(s)

        if now[0]!='people':
            if now[0] != n[0] or now[1] != n[1]:
                ls.append(np.array(m))
                m = []
        m.append([ float(i) for i in s])
        now[0] = n[0]
        now[1] = n[1]
    ls.append(np.array(m))
    return ls
