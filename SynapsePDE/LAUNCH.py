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

path_to_working_dir = 'paste the <PATH>'
SOL_STEP = 100 # [usec] initial time step for the PDE solution routine (adapts during calculation)
TSTEP = 2      # [N*SOL_STEP] the step for the mutual information calculation (aliquot to the solution step)
TDUR  = 10000  # total number of lag steps 

rA = syn_border;                        print('rA: ', rA*dr, ' nm') # the distance from the synapse center to the edge of the synapse
rB = int((pm_border + syn_border)/2);   print('rB: ', rB*dr, ' nm') # the distance from the synapse center to the center of the porous medium
rC = pm_border;                         print('rC: ', rC*dr, ' nm') # the distance from the synapse center to the edge of the porous medium


# writing solution to csv
for i in p.impls:
    for j in p.clearance_rates:
        sl = syn.sol_pde( impulses_number = i, uptake_rate = j )
        s  = sl.sol_ode()
        s.to_csv(path_to_working_dir + f'/solutions/SOL_imp{i}_clearance{j}.csv', encoding='utf-8', index=True, mode = 'w', header=True)



# _______________ calculation of mutual information
# take solution for each frequency and clearance rate from csv file
def retrieveData(imp_number, clearance):
    x = pd.read_csv( path_to_working_dir + f'/solutions/SOL_imp{imp_number}_clearance{clearance}.csv' )
    x.drop(columns=["Unnamed: 0"], inplace = True)
    return x


