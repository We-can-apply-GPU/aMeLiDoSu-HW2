import numpy as np
PHONES = 48
FBANKS = 69
import theano.tensor as T

def charto48(c):
    a = 0
    chrmap = open('data/48_idx_chr.map','r')
    for line in chrmap:
        s = line.split()
        for line in s:
            if s[0] == c:
                a = int(s[1])
                break
    return a

def read_examples(filename, sparm):
    ark = open('data/fbank/' + filename + '.ark','r')
    lab = open('data/label/' + filename + '.lab','r')
    datum = []
    #print("JJ{}".format(np.ones(5)))
    curPos = 0
    seqDic = {}

    for line in ark:
        s = line.rstrip().split(' ')
        for line in lab:
            l = line.rstrip().split(',')
            if s[0] == l[0]:        #map label to train data
                for i in range(1,len(s)):
                    s[i] = float(s[i])
                seqs = s[0].rstrip().split('_')
                s[0] = seqs[0] + seqs[1]
                s.append(l[1])
                datum.append(s) 
                curPos += 1
                break               
    #until now , datum is the list of [ID+frame  FBANKfeature phone]
    for i in range(len(datum)):
        #element = (datum[i][1],datum[i][2])
        if(datum[i][0] not in seqDic):
            seqDic[datum[i][0]] = ([],[])
        #seqDic[datum[i][0]][0].append(datum[i][1])
        seqDic[datum[i][0]][0].append([float(datum[i][k]) for k in range(1,len(datum[i])-1)])
        #seqDic[datum[i][0]][0] = np.array(seqDic[datum[i][0]][0])
        #print("test {}".format(seqDic[datum[i][0]][0].type))
        seqDic[datum[i][0]][1].append(charto48(datum[i][-1]))
    ans = []
    for key in seqDic:
        ans.append((np.array(seqDic[key][0]),np.array(seqDic[key][1])))
    return ans

def init_model(sample, sm, sparm):
    sm.size_psi = (PHONES + FBANKS) * PHONES  #48*48 + 69 * 48

def viterbi(x, w, y = []):
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
                tmp = np.zeros(PHONES)
                for pre in range(PHONES):
                    tmp[pre] = prob_pre[pre] + np.dot(w[FBANKS * now : FBANKS * (now+1)], xx) + w[FBANKS * PHONES + pre * PHONES + now]
                max_index = np.argmax(tmp)
                prob_now[now] = tmp[max_index]
                if len(y) != 0:
                    if now != y[index]:
                        prob_now[now] += 1
                each_trace[now] = max_index
        for i in range(len(prob_pre)):
            prob_pre[i] = prob_now[i]

        #print prob_pre == prob_now
        #print prob_pre is prob_now
        index += 1

    ans = [0] * len(x)
    ans[-1] = np.argmax(prob_pre)
    for i in range(1, len(x)):
        ans[-i-1] = trace[-i][ans[-i]]

    return ans


def classify_example(x, sm, sparm):
    return viterbi(x = x, w = sm.w)

def find_most_violated_constraint(x, y, sm, sparm):
    return viterbi(x = x, y = y, w = sm.w)

def psi(x, y, sm, sparm):
    import svmapi
    ###IMPORTANT###
    # (x,y) must be a value in seqDic!!
    feature = np.zeros(sm.size_psi)
    #feature = [0.0 for i in range(sm.size_psi)]
    for i in range(len(y) -1 ):   #y must be the same
        num1 = y[i]
        num2 = y[i+1]
        feature[FBANKS*PHONES + PHONES*num1 + num2] += 1
        for j in range(FBANKS):
            feature[num1*FBANKS+j] += x[i][j]
    return svmapi.Sparse(feature)

def loss(y, ybar, sparm):
    #print y, ybar
    cnt = 0
    for i, j in zip(y, ybar):
        if i != j:
            cnt += 1
    return float(cnt) / len(ybar)

def print_iteration_stats(ceps, cached_constraint, sample, sm,
                          cset, alpha, sparm):
    """Called just before the end of each cutting plane iteration.

    This is called just before the end of each cutting plane
    iteration, primarily to print statistics.  The 'ceps' argument is
    how much the most violated constraint was violated by.  The
    'cached_constraint' argument is true if this constraint was
    constructed from the cache.
    
    The default behavior is that nothing is printed."""
    print

def print_learning_stats(sample, sm, cset, alpha, sparm):
    """Print statistics once learning has finished.
    
    This is called after training primarily to compute and print any
    statistics regarding the learning (e.g., training error) of the
    model on the training sample.  You may also use it to make final
    changes to sm before it is written out to a file.  For example, if
    you defined any non-pickle-able attributes in sm, this is a good
    time to turn them into a pickle-able object before it is written
    out.  Also passed in is the set of constraints cset as a sequence
    of (left-hand-side, right-hand-side) two-element tuples, and an
    alpha of the same length holding the Lagrange multipliers for each
    constraint.

    The default behavior is that nothing is printed."""
    print 'Model learned:',
    print '[',', '.join(['%g'%i for i in sm.w]),']'
    print 'Losses:',
    print [loss(y, classify_example(x, sm, sparm), sparm) for x,y in sample]

def print_testing_stats(sample, sm, sparm, teststats):
    """Print statistics once classification has finished.
    
    This is called after all test predictions are made to allow the
    display of any summary statistics that have been accumulated in
    the teststats object through use of the eval_prediction function.

    The default behavior is that nothing is printed."""
    print teststats

def eval_prediction(exnum, (x, y), ypred, sm, sparm, teststats):
    """Accumulate statistics about a single training example.
    
    Allows accumulated statistics regarding how well the predicted
    label ypred for pattern x matches the true label y.  The first
    time this function is called teststats is None.  This function's
    return value will be passed along to the next call to
    eval_prediction.  After all test predictions are made, the last
    value returned will be passed along to print_testing_stats.

    On the first call, that is, when exnum==0, teststats==None.  The
    default behavior is that the function does nothing."""
    if exnum==0: teststats = []
    print 'on example',exnum,'predicted',ypred,'where correct is',y
    teststats.append(loss(y, ypred, sparm))
    return teststats

def write_model(filename, sm, sparm):
    """Dump the structmodel sm to a file.
    
    Write the structmodel sm to a file at path filename.

    The default behavior is equivalent to
    'cPickle.dump(sm,bz2.BZ2File(filename,'w'))'."""
    import cPickle, bz2
    f = bz2.BZ2File("model/" + filename, 'w')
    cPickle.dump(sm, f)
    f.close()

def read_model(filename, sparm):
    """Load the structure model from a file.
    
    Return the structmodel stored in the file at path filename, or
    None if the file could not be read for some reason.

    The default behavior is equivalent to
    'return cPickle.load(bz2.BZ2File(filename))'."""
    import cPickle, bz2
    return cPickle.load(bz2.BZ2File(filename))

def write_label(fileptr, y):
    """Write a predicted label to an open file.

    Called during classification, this function is called for every
    example in the input test file.  In the default behavior, the
    label is written to the already open fileptr.  (Note that this
    object is a file, not a string.  Attempts to close the file are
    ignored.)  The default behavior is equivalent to
    'print>>fileptr,y'"""
    print>>fileptr,y
