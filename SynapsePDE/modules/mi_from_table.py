import math
import numpy as np
import pandas as pd
pd.options.display.float_format = '{:.2e}'.format
import matplotlib.pyplot as plt

def preprocess_table(path_to_table):
    x = pd.read_csv(path_to_table)
    x.rename(columns={"Unnamed: 0": "usec"}, inplace = True)
    x.set_index( "usec", inplace = True )
    return x

def joint_prob_table( data1, data2 ):
    a = list(data1); b = list(data2)
    M = np.zeros( (len(a), len(b)) ).astype(float)
    for i in range(len(a)):
        for j in range(len(b)):
            M[i,j] = max(0., a[i] + b[j])
    M = M/sum(sum(M))
    return M

def joint_count( data1, loc1, data2, loc2 ):
    a = list(data1); b = list(data2)
    M = np.zeros( (len(a), len(b)) ).astype(int)
    for i in range(len(a)):
        for j in range(len(b)):
            M[i,j] = max(0., a[i] + b[j])
    return M[loc1,loc2]

def condEntr(Y,X, data):
    """ H(Y|X) = sum_xy{ p(x,y) log2( p(x,y)/p(x) ) } """
    if data.columns[0] == 'Unnamed: 0':
        data.drop(columns = ['Unnamed: 0'], inplace = True)

    ce = []
    for t in range(data.shape[0]):
        jd = joint_prob_table( data.iloc[t, :], data.iloc[t, :] )
        px = abs(data.iloc[t,X])/np.sum( abs(data.iloc[:, X]) )
        if jd[X,Y] == 0. or px == 0.:
            k = 0
        else:
            k = abs( jd[X,Y] * math.log( abs(jd[X,Y]) / px ))
        ce.append(k)
    return pd.DataFrame( ce, index = data.index, columns = [str(Y)+'|'+str(X)] )

def mi_vals(lA, lB, data):
    mmi = []
    Tstep = 20
    Tdur  = 4000
    for i in np.arange(start = 0, stop = Tdur, step = Tstep):
        m = 0
        for t in range(data.shape[0]-i-1):
            jd = joint_prob_table( data.iloc[t,:], data.iloc[t+i,:] )
            margProbA = np.sum(jd,axis = 0); margProbB = np.sum(jd,axis = 1)
            if jd[lA,lB] == 0. or margProbA[lA] == 0. or margProbB[lB] == 0.:
                mi = 0
            else:
                mi = jd[lA,lB] * abs(math.log( jd[lA,lB]/margProbA[lA]/margProbB[lB], 2 ))
            m += mi
        print(i)
        mmi.append(m)
    return pd.DataFrame( mmi, index = np.arange(start = 0, stop = Tdur, step = Tstep) )

def mi_from_np(lA, lB, data):
    mi = []
    Tstep = 100
    Tdur  = 3000
    for T in np.arange(start = 0, stop = Tdur, step = Tstep):
        m = 0
        for t in range(data.shape[0]-T-1):
            nA  = data.iloc[t,:]
            nB  = data.iloc[t+T,:]
            joint_prob, edges = np.histogramdd([ nA, nB ], bins = data.shape[1], density = True); joint_prob /= joint_prob.sum()
            margA = joint_prob.sum(axis=0)
            margB = joint_prob.sum(axis=1)
            if joint_prob[ lA, lB ] == 0. or margA[lA] == 0. or margB[lB] == 0.:
                x = 0
            else:
                x = abs(math.log( joint_prob[ lA, lB ]/margA[lA]/margB[lB] , 2))
            m += round(joint_prob[ lA, lB ] * x, 2)
        print(T)
        mi.append(m)
    return pd.DataFrame( mi, index = np.arange(start = 0, stop = Tdur, step = Tstep) )

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
