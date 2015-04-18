import numpy as np
index2char = {}
char2index = {}
index2ans = {}
a = 0
chrmap = open('data/48_idx_chr.map','r')
for line in chrmap:
    line = line.split()
    index2char[int(line[1])] = line[0]
    char2index[line[0]] = int(line[1])
    index2ans[int(line[1])] = line[2]
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
    lines.sort(key=lambda line: line[0][2])
    ans = {}
    for line in lines:
        if not line[0][0] in ans:
            ans[line[0][0]] = {}
        if not line[0][1] in ans[line[0][0]]:
            ans[line[0][0]][line[0][1]] = []
        ans[line[0][0]][line[0][1]] += [line[1:]]
    return ans

def read_label(filename):
    fin = open("data/label/" + filename + ".lab", "r")
    lines = []
    for line in fin:
        line = line.rstrip().split(',')
        line[0] = line[0].split('_')
        line[0][2] = int(line[0][2])
        lines += [line]
    lines.sort(key=lambda line: line[0][2])
    ans = {}
    for line in lines:
        if not ans.has_key(line[0][0]):
            ans[line[0][0]] = {}
        if not ans[line[0][0]].has_key(line[0][1]):
            ans[line[0][0]][line[0][1]] = []
        ans[line[0][0]][line[0][1]] += [line[1]]
    return ans

def read_weight(filename):
    import json
    f = open("model/" + filename, "r")
    return json.loads(f.readline())

def viterbi(x, w, y = []):
    w = list(w)
    lenx = len(x)
    x = x.reshape((lenx, FBANKS))
    observation = np.array(w[:PHONES*FBANKS]).reshape((PHONES, FBANKS))
    trans = np.array(w[PHONES*FBANKS:]).reshape((PHONES, PHONES))
    xobs = np.dot(x, observation.T)
    """
    print x.shape
    print observation.T.shape
    print trans.shape
    print xobs.shape
    """
    prob_pre = np.zeros((PHONES, 1))
    trace = []
    for i in range(lenx):
        prob_now = prob_pre + trans + xobs[i, :]
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
    """
    prob_pre = np.zeros(PHONES)
    prob_now = np.zeros(PHONES)
    trace = np.zeros((len(x), PHONES), dtype=np.int16) 
    index = 0
    for xx, each_trace in zip(x, trace):
        for now in range(PHONES):
            if index == 0:
                prob_now[now] = np.dot(w[FBANKS * now : FBANKS * (now+1)], xx[:])
                if len(y) != 0:
                    if now != y[index]:
                        prob_now[now] += 1
            else:
                ary = np.zeros(PHONES)
                tmp = np.dot(w[FBANKS * now : FBANKS * (now+1)], xx)
                for pre in range(PHONES):
                    ary[pre] = prob_pre[pre] + tmp + w[FBANKS * PHONES + pre * PHONES + now]
                max_index = np.argmax(ary)
                prob_now[now] = ary[max_index]
                if len(y) != 0:
                    if now != y[index]:
                        prob_now[now] += 1
                each_trace[now] = max_index
        for i in range(len(prob_pre)):
            prob_pre[i] = prob_now[i]

        index += 1

    ans = [0] * len(x)
    ans[-1] = np.argmax(prob_pre)
    for i in range(1, len(x)):
        ans[-i-1] = trace[-i][ans[-i]]

    return ans
    """
