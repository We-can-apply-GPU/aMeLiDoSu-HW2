import numpy as np
import sys
sys.path.append("userCode")
import util
PHONES = 48
FBANKS = 69
MAX_SPEAKER = 1
def read_examples(filename, sparm):
    fbank = util.read_fbank(filename)
    print "Reading label..."
    label = util.read_label(filename)
    print "Processing..."
    print len(label)
    ans = []
    for (i, (speaker_id, v)) in enumerate(label.iteritems()):
        for (sequence_id, content) in v.iteritems():
            content = [util.char2index[x] for x in content]
            ans.append((np.array(fbank[speaker_id][sequence_id]), np.array(content)))
        if i == MAX_SPEAKER:
            break
    return ans

def init_model(sample, sm, sparm):
    sm.size_psi = (PHONES + FBANKS) * PHONES

def classify_example(x, sm, sparm):
    return util.viterbi(x = x, w = sm.w)

def find_most_violated_constraint(x, y, sm, sparm):
    return util.viterbiDelta(x = x, y = y, w = sm.w)

def psi(x, y, sm, sparm):
    import svmapi
    ###IMPORTANT###
    # (x,y) must be a value in seqDic!!
    #feature = np.zeros(sm.size_psi)
    feature = [0.0 for i in range(sm.size_psi)]
    for i in range(len(y) -1 ):   #y must be the same
        num1 = y[i]
        num2 = y[i+1]
        feature[FBANKS*PHONES + PHONES*num1 + num2] += 1
        for j in range(FBANKS):
            feature[num1*FBANKS+j] += x[i][j]
    return svmapi.Sparse(feature)

def loss(y, ybar, sparm):
    #print y, ybar
    #cnt = 0
    #for i, j in zip(y, ybar):
        #if i != j:
            #cnt += 1
    #return float(cnt)/len(y)
    
    cnt = 0
    yl = ''
    ybl = ''
    for i in range(len(y)):
        alpha = (y[i] != yl)
        beta = (ybar[i] != ybl)
        if alpha:
            cnt += 1
            yl = y[i]
        if beta:
            cnt += 1
            ybl = ybar[i]
        if (alpha or beta) and (y[i] == ybar[i]):
            cnt -= 2
    return cnt

def print_learning_stats(sample, sm, cset, alpha, sparm):
    print 'Losses:',
    for x, y in sample:
        ybar = classify_example(x, sm, sparm)
        print y
        print ybar
        print loss(y, ybar, sparm)

def write_model(filename, sm, sparm):
    import cPickle, bz2, json
    f = bz2.BZ2File("SVMmodel/" + filename, 'w')
    cPickle.dump(sm, f)
    f.close()
    with open("model/" + filename,'w') as weight:
        weight.write(json.dumps(list(sm.w)))

