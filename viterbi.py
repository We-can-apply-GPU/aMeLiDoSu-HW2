import numpy as np

def viterbi(ob,tsm,opm,st):
    ol = ob.__len__()
    gl = opm.shape[0]
    pl = opm.shape[1]
    rs = [[0]*ol for i in range(gl)]
    for i in range(gl):
        rs[i][0] = (-1,st[i]*opm[i][ob[0]])
    for k in range(1,ol):
        for i in range(gl):
            mx = 0
            ix = -1
            for j in range(gl):
                if rs[j][k-1][1]*opm[i][ob[k]]*tsm[j][i] > mx:
                    mx = rs[j][k-1][1]*opm[i][ob[k]]*tsm[j][i]
                    ix = j
            rs[i][k] = (ix,mx)
    trc = [0]*ol
    for i in range(gl):
        mx = 0
        if rs[i][-1][1] > mx:
            mx = rs[i][-1][1]
            trc[-1] = i
    for i in range(1,ol):
        trc[-i-1] = rs[trc[-i]][-i][0]
    print(rs)
    return trc

if __name__ == '__main__':
    ob = [0,1,2]
    st = [0.6,0.4]
    tsm = np.asarray([[0.7,0.3],[0.4,0.6]])
    opm = np.asarray([[0.5,0.4,0.1],[0.1,0.3,0.6]])
    trc = viterbi(ob,tsm,opm,st)
    print(trc)
