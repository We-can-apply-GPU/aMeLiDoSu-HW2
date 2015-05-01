import numpy as np
def hw1data(f = 'data'):
    d = open(f)
    m = []
    for i in d:
        s = (i.rstrip().split())[1]
        s = (s[1:-1]).split(',')
        print(s)
        m.append([ float(i) for i in s])
    return np.array(m)
