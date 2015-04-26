import numpy as np
import sys
sys.path.append("userCode")
import util
PHONES = 48
FBANKS = 69

def read_examples(filename, sparm):
    fbank = util.read_fbank(filename)
    print "Reading label..."
    label = util.read_label(filename)
    print "Processing..."
    print len(label)
    ftmp = []
    ltmp = []
    ans = []
    for f, l in zip(fbank, label):
        if f[0][2] == 1 and len(ftmp) != 0:
            ans.append((np.array(ftmp), np.array(ltmp)))
            ftmp = []
            ltmp = []
        ftmp += [f[1:]]
        ltmp += [util.char2index[l[1]]]
    if len(ftmp) != 0:
        ans.append((np.array(ftmp), np.array(ltmp)))
    return ans

def init_model(sample, sm, sparm):
    sm.size_psi = (PHONES + FBANKS) * PHONES

def classify_example(x, sm, sparm):
    return util.viterbi(x = x, w = sm.w)

def find_most_violated_constraint_margin(x, y, sm, sparm):
    return util.viterbi(x = x, y = y, w = sm.w)

def psi(x, y, sm, sparm):
    import svmapi
    feature = np.zeros((sm.size_psi, 1))
    for i in range(len(y)):   #y must be the same
        num1 = y[i]
        feature[num1*FBANKS:(num1+1)*FBANKS] += x[i].reshape((FBANKS, 1))
        if i != len(y)-1:
            num2 = y[i+1]
            feature[FBANKS*PHONES + PHONES*num1 + num2] += 1
    return svmapi.Sparse(np.array(feature))

def loss(y, ybar, sparm):
    cnt = 0
    for i, j in zip(y, ybar):
        if i != j:
            cnt += 1
    return float(cnt)/len(y)

def print_learning_stats(sample, sm, cset, alpha, sparm):
    print 'Losses:',
    for x, y in sample:
        ybar = classify_example(x, sm, sparm)
        print loss(y, ybar, sparm)

def write_model(filename, sm, sparm):
    import cPickle, bz2, json
    f = bz2.BZ2File("SVMmodel/" + filename, 'w')
    cPickle.dump(sm, f)
    f.close()
    with open("model/" + filename,'w') as weight:
        weight.write(json.dumps(list(sm.w)))

