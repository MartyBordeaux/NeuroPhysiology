import numpy as np
import pandas as pd
pd.options.display.float_format = '{:.2e}'.format
import matplotlib.pyplot as plt
import timeit

import modules.pars_synapse as p
import modules.pde_synapse as syn
import modules.mi_from_table as m

# HYPERPARAMETERS
dr = p.diffusion['R_syn']/100 # step = 10 nm
syn_border = int(p.diffusion['synapse_zone']/dr)
pm_border = int(p.diffusion['transition_zone']/dr)

rA = syn_border;                        print('rA: ', rA*dr, ' nm')
rB = int((pm_border + syn_border)/2);   print('rB: ', rB*dr, ' nm')
rC = pm_border;                         print('rC: ', rC*dr, ' nm')

impls = [3, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32]
freqs = np.array(impls)/stim_duration # kHz
clearance_rates = [0.0, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]

# writing solution to csv
for i in impls:
    for j in clearance_rates:
        sl = syn.sol_pde( impulses_number = i, uptake_rate = j, delay = 0 )
        s  = sl.sol_ode()
        s.to_csv(f'/<PATH>/SOL_imp{i}_diss{j}_delay0.csv', encoding='utf-8', index=True, mode = 'w', header=True)



# _______________ calculation of mutual information
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def retrieveData(imp_number, clearance):
    x = pd.read_csv( f'/<PATH>/SOL_imp{imp_number}_diss{clearance}_delay0.csv' )
    x.drop(columns=["Unnamed: 0"], inplace = True)
    return x

# writing MI to csv
for i in impls:
    for j in clearance_rates:
        sl = m.mi_vals(rA, rB, retrieveData( i, j ) )
        sl.to_csv(f'/<PATH>/MI_imp{i}_diss{j}_delay0.csv', encoding='utf-8', index=True, mode = 'w', header=True)

