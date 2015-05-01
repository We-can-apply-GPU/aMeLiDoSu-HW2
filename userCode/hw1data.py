import numpy as np
def hw1data(f = 'data'):
    d = open(f)
    now=['people','talk']
    ls = []
    m = []
    for i in d:
        s = i.rstrip().split(',[')
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
