# python3 /home/shuvaev/MEGA/codes/python/SynapsePDE/2024.07.09_miProcessing.py

import os
import pandas as pd
import numpy as np



impls = [1,5,10,50,100]
clrnc = [.0, .001, .005, .01, .05, .1, .5]






class processingMI:
    def __init__(self, path_to_solutions, path_to_MI_folder):
        self.pathMI = path_to_MI_folder
        self.pathSL = path_to_solutions




    def retrieveSolution(imp_number, clearance, solutionStep):
        try:
            x = pd.read_csv( path_to_archive + f'sol/SOL{solutionStep}_imp{imp_number/100}_clrnc{clearance}.csv' )
            x.drop(columns=["Unnamed: 0"], inplace = True)
        except FileNotFoundError:
            print(f'No solution for {imp_number} impulses and {clearance} clearance')
        return x




    def calculateMI(self, TDUR, TSTEP, saveTable=False):
        Tstep = TSTEP
        Tdur  = TDUR

        def joint_prob_table( data1, data2 ):
            a = list(data1); b = list(data2)
            M = np.zeros( (len(a), len(b)) ).astype(float)
            for i in range(len(a)):
                for j in range(len(b)):
                    M[i,j] = max(0., a[i] + b[j])
            M = M/sum(sum(M))
            return M
        def mi_vals(lA, lB, data):
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

        for imp in tqdm(impls):
            df = pd.DataFrame({'lag': np.arange(0, TDUR, TSTEP )})
            try:
                pd.read_csv( self.pathMI + f'MI{TSTEP}_imp{imp/100}_0rB.csv' )
            except FileNotFoundError:
                for cl in clrnc:
                    try:
                        x = pd.read_csv( self.pathSL + f'SOL{SOL_STEP}_imp{imp/100}_diss{cl}.csv' )
                        mi_tab = mi_vals(0, rB, retrieveData(imp,  cl, SOL_STEP))
                        df[str(cl)] = mi_tab.values
                    except FileNotFoundError:
                        pass
            df.to_csv(self.pathMI + f'MI{TSTEP}_imp{imp/100}_0rB.csv', encoding='utf-8', index=False, mode = 'w', header=True)




    def createLong(self, saveTable = False):
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




    def calculateMaxMI(self, saveTable = False):
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

    def merge_Hz_kHz_in_0rB(self, saveTable = False):
        Hz = processingMI.createMax(self, saveTable = False)
        kHz = pd.read_csv(self.pathMI + 'ALL_MI_max.csv')
        kHz = kHz[ kHz['lctn'] == '0rB' ]
        kHz = kHz[['MI', 'imps','diss', 'freqs']]
        kHz.rename({'MI': 'maxVal', 'imps': 'imps', 'diss': 'clrnc', 'freqs': 'freq'}, axis=1, inplace=True)
        kHz['freq'] = kHz['freq'].values * 1000
        dfFin = pd.concat( [Hz, kHz] )
        if saveTable:
            dfFin.to_csv(self.pathMI + 'Final_maxMI.csv', encoding='utf-8', index=True, mode = 'w', header=True)
        return dfFin


X = processingMI(path_to_solutions = '/home/shuvaev/MEGA/vrem/calcs/sols/', path_to_MI_folder = '/home/shuvaev/MEGA/vrem/calcs/mi/')
X.createLong(saveTable = True)
X.calculateMaxMI(saveTable = True)

#print(X.merge_Hz_kHz_in_0rB(saveTable=False))
