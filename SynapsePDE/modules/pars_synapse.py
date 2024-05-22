# [1] Barbour, 2001
# [2] Holmes, 1995

import numpy as np


# __________ parameters dictionary
diffusion = {
    'D_cleft':          7.60E+02,   # [nm^2/usec]   diffusion inside synapse [1] ( 7.60E-06 cm2/sec [1]) = 7.60E+08 nm2/sec = 7.60E+02 nm2/usec = 7.60E-04 um2/usec
    'D_pm':             2.97E+02,   # [nm^2/usec]   diffusion inside porous medium [1] ( 2.97E-06 cm2/sec [1]) = 2.97E+08 nm2/sec = 2.97E+02 nm2/usec = 2.97E-04 um2/usec
    'R_syn':            1.00E+03,   # [nm]          synapse radius
    'synapse_zone':     2.00E+02,   # [nm]          emission zone radius (synapse outer radius)
    'transition_zone':  4.00E+02,   # [nm]          transition to porous medium outer radius
    'h_syn':            2.00E+01,   # [nm]          synapse height [1]
    'alpha_syn':        2.00E-01,   # [-]           volume fraction of brain tissue [1]
    'U_cleft':          1.00E-01,   # [nm/usec]     glutamate uptake rate [2] (80 molecules/ms ???)
    'U_pm':             1.00E+00,   # [nm/usec]     glutamate uptake rate [2] (80 molecules/ms ???)
    'tau_diss':         1.00E+03,   # [usec]        glu uptake time constant
}


# __________ for coordinates recalculation
theta = np.linspace(0, 2*np.pi, 100)
def polar2z(r,theta): return r * np.exp( 1j * theta )


# __________ polynomial interpolation
import numpy.linalg as lin
a = diffusion['synapse_zone']
b = diffusion['transition_zone']

V = np.array([  [1., a,  a**2, a**3,   a**4,    a**5],
                [1., b,  b**2, b**3,   b**4,    b**5],
                [0., 1., 2*a,  3*a**2, 4*a**3,  5*a**4],
                [0., 1., 2*b,  3*b**2, 4*b**3,  5*b**4],
                [0., 0., 2.,   6*a,    12*a**2, 20*a**3],
                [0., 0., 2.,   6*b,    12*b**2, 20*b**3] ])

y = np.array([0., 1., 0., 0., 0., 0.])
xi_s = lin.solve(V,y)

def f_inter(r):
    s = 0
    for i in range(len(xi_s)):
        s += xi_s[i]*r**i
    return s

# __________ stimulation in our experiment
record_time = 5.00E+06 # [usec]
stim_start  = 1.00E+04 # [usec]
stim_end    = 1.32E+05 # [usec]
