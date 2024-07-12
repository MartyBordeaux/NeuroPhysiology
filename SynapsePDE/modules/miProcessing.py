# python3 /home/shuvaev/MEGA/codes/python/SynapsePDE/2024.07.09_miProcessing.py

import os
import pandas as pd
import numpy as np

import pars_synapse as p

""" This module calculates the mutual information from the tables of solutions, which are supposed to be in the seperate folder """




class processingMI:
    def __init__(self, path_to_solutions, path_to_MI_folder):
        self.pathMI = path_to_MI_folder # wher to store the results of MI calculation
        self.pathSL = path_to_solutions # the source folder with the solution of PDE tables

    def retrieveSolution(imp_number, clearance, solutionStep):
        try:
            x = pd.read_csv( self.pathSL + f'sol/SOL{solutionStep}_imp{imp_number/100}_clrnc{clearance}.csv' )
            x.drop(columns=["Unnamed: 0"], inplace = True)
        except FileNotFoundError:
            print(f'No solution for {imp_number} impulses and {clearance} clearance')
        return x

    def calculateMI(self, TDUR, TSTEP, saveTable=False):
        Tstep = 2 # can be changed
        Tdur  = 10000

        def joint_prob_table( data1, data2 ): # calculation of joint probabiliyies
            a = list(data1); b = list(data2)
            M = np.zeros( (len(a), len(b)) ).astype(float)
            for i in range(len(a)):
                for j in range(len(b)):
                    M[i,j] = max(0., a[i] + b[j])
            M = M/sum(sum(M))
            return M
        def mi_vals(lA, lB, data): # THIS CALCULATES THE MUTUAL INFORMATION
            mmi = []
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
                #print(i)
                mmi.append(m)
            return pd.DataFrame( mmi, index = np.arange(start = 0, stop = Tdur, step = Tstep) )

        for imp in tqdm(p.impls):
            df = pd.DataFrame({'lag': np.arange(0, TDUR, TSTEP )})
            try:
                pd.read_csv( self.pathMI + f'MI{TSTEP}_imp{imp/100}_0rB.csv' )
            except FileNotFoundError:
                for cl in p.clrnc:
                    try:
                        x = pd.read_csv( self.pathSL + f'SOL{SOL_STEP}_imp{imp/100}_clearance{cl}.csv' )
                        mi_tab = mi_vals(0, rB, retrieveData(imp,  cl, SOL_STEP))
                        df[str(cl)] = mi_tab.values
                    except FileNotFoundError:
                        pass
            df.to_csv(self.pathMI + f'MI{TSTEP}_imp{imp/100}_0rB.csv', encoding='utf-8', index=False, mode = 'w', header=True)




    def createLong(self, saveTable = False): # reshaping the mutual information tables and combining all calculations together
        df = pd.DataFrame({'lag':[], 'clrnc': [], 'value': [], 'imps': [], 'lctn': []})
        miFiles = os.listdir(self.pathMI)
        miFiles = [value for value in miFiles if value != 'ALL_MI_Hz.csv']
        miFiles = [value for value in miFiles if value != 'ALL_maxMI_Hz.csv']
        for files in miFiles:
            imps = files.split("_")[1][3:]
            mi = os.path.join(self.pathMI, files)
            x  = pd.read_csv( mi )
            colNames = list(x.columns)
            y = pd.melt(x, id_vars='lag', value_vars=colNames, var_name='clrnc', value_name='value')
            y['imps'] = np.repeat(imps, y.shape[0])
            y['lctn'] = np.repeat('0rB', y.shape[0])
            df = pd.concat( [df, y])
        if saveTable:
            df.to_csv(self.pathMI + 'ALL_MI_Hz.csv', encoding='utf-8', index=False, mode = 'w', header=True)
        return df

    def calculateMaxMI(self, saveTable = False): # calculation of maximal values of MI
        inDF = processingMI.createLong(self, saveTable = False)
        imp = np.unique(inDF['imps'].values.astype(float))
        clr = np.unique(inDF['clrnc'].values.astype(float))
        maxD = []
        for i in imp:
            for j in clr:
                x = inDF[ inDF['imps'] == i.astype(str) ]
                x = x[ x['clrnc'] == j.astype(str)]
                try:
                    y = max(x['value'])
                    maxD.append([y, i , float(j) ])
                except ValueError:
                    pass
        maxMI = pd.DataFrame(maxD, columns = ['maxVal', 'imps', 'clrnc'])
        maxMI['freq'] = maxMI['imps']/2*1000
        if saveTable:
            maxMI.to_csv(self.pathMI + 'ALL_maxMI_Hz.csv', encoding='utf-8', index=False, mode = 'w', header=True)
        return maxMI
