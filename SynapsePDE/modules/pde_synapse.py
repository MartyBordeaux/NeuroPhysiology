import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, ode, odeint
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from scipy import optimize
from scipy.stats import norm
import ddeint

from matplotlib.animation import FuncAnimation

import modules.pars_synapse as p


class sol_pde:
    def __init__(self, impulses_number, uptake_rate, delay):
        self.impulses_number    = impulses_number
        self.uptake_rate        = uptake_rate
        self.delay              = delay
        self.dr         = p.diffusion['R_syn']/100                      # size of the cell
        self.num_rings  = int(p.diffusion['R_syn']/self.dr)             # number of all compartments
        self.syn_border = int(p.diffusion['synapse_zone']/self.dr)      # position of outer synapse border
        self.prm_border = int(p.diffusion['transition_zone']/self.dr)   # position of inner porous medium border

        self.t0 = 0.00E+00; self.tfin = 5E4 #p.record_time-p.stim_start;
        self.stp = 1 # (step = 1 usec)
        
        if self.impulses_number > 1:
            imp_step = 2E3 // self.impulses_number / self.stp #p.stim_end
            self.impulses = [ self.stp*i*imp_step for i in range(self.impulses_number) ]
            t1 = np.arange(start = self.t0, stop = self.stp*self.impulses_number*imp_step + self.stp, step = self.stp)
            t2 = np.arange(start = self.stp*self.impulses_number*imp_step + 2*self.stp, stop = int(self.tfin/2), step = 20*self.stp)
            t3 = np.arange(start = int(self.tfin/2) + self.stp, stop = self.tfin, step = 100*self.stp)
            self.t = np.append(t1, t2)
            self.t = np.append(self.t, t3)
        else:
            t1 = np.arange(start = self.t0, stop = int(self.tfin/10), step = self.stp)
            t2 = np.arange(start = int(self.tfin/10) + self.stp, stop = self.tfin, step = 5*self.stp)
            self.t = np.append(t1, t2)


    def init_boundaries(self):
        InitialCondition = lambda t: 1E9 if t == self.t0 else 0
        y0 = np.zeros(self.num_rings)
        for ri in range( int(p.diffusion['synapse_zone']/self.dr) ):
            y0[ri] = int(InitialCondition(0) * norm.pdf( ri, 0, self.syn_border * .3 ))
        return y0


    def rhs_system(self, t, y):
        def DiracNormal(t, t0, sgm): return np.exp(-((t-t0)/sgm)**2)/sgm/1.77
        def DiracNormalPrime(t, t0, sgm): return DiracNormal(t,t0,sgm) * 2/sgm**2 * ( t0 - t )

        def impls(t, loc):
            if self.impulses_number > 1:
                for j in range(1,len(self.impulses)):
                    if t-self.stp >= self.impulses[j] - self.stp and t < self.impulses[j] + self.stp:
                        r = DiracNormalPrime(t, self.impulses[j], 0.1) * int(1E9 * norm.pdf( loc, 0, self.syn_border * .3 ))
                        break
                    else:
                        r = 0
            else:
                r = 0
            return r


        def Diff(r):
            if r < self.syn_border:
                out = p.diffusion['D_cleft']
            elif r >= self.syn_border and r <= self.prm_border:
                out = p.diffusion['D_cleft'] + p.f_inter(r*self.dr)*( p.diffusion['D_pm'] - p.diffusion['D_cleft'] )
            else:
                out = p.diffusion['D_pm']
            return out
        def Volm(r):
            if r < self.syn_border:
                out = np.pi*p.diffusion['h_syn']*(r*self.dr)**2
            elif r >= self.syn_border and r <= self.prm_border:
                out = np.pi*p.diffusion['h_syn']*(r*self.dr)**2 + p.f_inter(r*self.dr)*( p.diffusion['alpha_syn']*4/3*np.pi*(r*self.dr)**3 - np.pi*p.diffusion['h_syn']*(r*self.dr)**2 )
            else:
                out = p.diffusion['alpha_syn']*4/3*np.pi*(r*self.dr)**3
            return out
        def vol_coeff(r): return (Volm(r+1) - 2*Volm(r) + Volm(r-1))/(Volm(r+1) - Volm(r-1))*2/self.dr

        def uptk(r):
            if r < self.syn_border:
                out = self.uptake_rate #p.diffusion['U_cleft']
            elif r >= self.syn_border and r <= self.prm_border:
                out = self.uptake_rate + p.f_inter(r*self.dr)*( 3*self.uptake_rate - self.uptake_rate )
            else:
                out = 3*self.uptake_rate # p.diffusion['U_pm']
            return out

        def exp_diss(t):
            if self.impulses_number > 1:
                for j in range(len(self.impulses)-1):
                    if t >= self.impulses[j] and t < self.impulses[j+1]:
                        timp = self.impulses[j]
                        break
                    else:
                        timp = 0
            else:
                timp = 0
            out = math.exp( -(t-timp)/p.diffusion['tau_diss'] ) if (t-timp) > 0 else 0
            return out
        def delayed_diss( t, r, delay): return uptk(r) * exp_diss(t - delay)

        dc = np.zeros(self.num_rings)
        # __________ solution inside synapse
        dc[0] = 4 * p.diffusion['D_cleft']/self.dr**2 * (y[1] - y[0]) -\
                2 * uptk(r=0) * (y[1] - y[0])/self.dr - impls(t, 0)
        for ri in range(1, self.num_rings-1):
            if ri < self.syn_border:
                dc[ri] = p.diffusion['D_cleft'] * ( y[ri+1] + y[ri-1] - 2*y[ri] )/self.dr**2 +\
                         (p.diffusion['D_cleft']/ri - uptk(r=ri))*(y[ri+1] - y[ri-1])/(2*self.dr) - uptk(r=ri)/ri * y[ri] - impls(t, ri)
            elif ri >= self.syn_border and ri <= self.prm_border:
                dc[ri] = Diff(ri) * ( y[ri+1] + y[ri-1] - 2*y[ri] )/self.dr**2 +\
                         ((Diff(ri+1) - Diff(ri-1))/(2*self.dr) + Diff(ri)*vol_coeff(ri) - uptk(r=ri)) * (y[ri+1] - y[ri-1])/(2*self.dr) -\
                         ( (uptk(r=ri+1) - uptk(r=ri-1))/(2*self.dr) + uptk(r=ri)*vol_coeff(ri))*y[ri]
            else:
                dc[ri] = p.diffusion['D_pm'] * ( y[ri+1] + y[ri-1] - 2*y[ri] )/self.dr**2 +\
                         ( 2*p.diffusion['alpha_syn']*p.diffusion['D_pm']/ri - uptk(r=ri))*(y[ri+1] - y[ri-1])/(2*self.dr) -\
                         ( (uptk(r=ri+1) - uptk(r=ri-1))/(2*self.dr) + uptk(r=ri)/ri)*y[ri]
        dc[self.num_rings-1] = p.diffusion['D_pm'] * ( y[self.num_rings-2] - 2*y[self.num_rings-1] )/self.dr**2 -\
                         (p.diffusion['D_pm']/self.num_rings - uptk(r=self.num_rings)) *  y[self.num_rings-1]/self.dr -\
                         uptk(r=self.num_rings)/self.num_rings * y[self.num_rings-1]
        return dc


    def sol_ode(self):
        inits = sol_pde.init_boundaries(self)
        s = ode(lambda t, y: sol_pde.rhs_system(self, t, y), jac=None).set_integrator(name = 'vode', method='bdf', with_jacobian=False)
        s.set_initial_value(inits, self.t0)
        sol = []; tm = []
        for i in tqdm(range(1,len(self.t))):
            s.integrate(self.t[i])
            sol.append(s.y)
            tm.append(s.t)
        #print(self.impulses)
        return pd.DataFrame(sol, index = tm)


    def plot_sol(self, mode):
        sltn    = sol_pde.sol_ode(self)
        if mode == 'quick':
            times = [int(len(sltn)/2.5), int(len(sltn)/2), int(len(sltn)/1.5), int(len(sltn)/1.1), int(len(sltn)-1)]
            r_r = self.num_rings-0
            plt.plot(range(r_r), sltn.iloc[times[0], 0:r_r].values, 'k-', alpha = .9, label = str(round(self.t[times[0]]/1000,1)),linewidth = 3.0)
            plt.plot(range(r_r), sltn.iloc[times[1], 0:r_r].values, 'k-', alpha = .8, label = str(round(self.t[times[1]]/1000,1)),linewidth = 3.0)
            plt.plot(range(r_r), sltn.iloc[times[2], 0:r_r].values, 'k-', alpha = .6, label = str(round(self.t[times[2]]/1000,1)),linewidth = 3.0)
            plt.plot(range(r_r), sltn.iloc[times[3], 0:r_r].values, 'k-', alpha = .4, label = str(round(self.t[times[3]]/1000,1)),linewidth = 3.0)
            plt.plot(range(r_r), sltn.iloc[times[4], 0:r_r].values, 'k-', alpha = .3, label = str(round(self.t[times[4]]/1000,1)),linewidth = 3.0)
            plt.legend(loc = 'best', title = 'Timepoints (msec)')
            plt.axvline( x = p.diffusion['synapse_zone']/self.dr, color='k', ls='--' )
            plt.axvline( x = p.diffusion['transition_zone']/self.dr, color='k', ls='--' )
        elif mode == 'dynamic1d':
            fig, ax = plt.subplots(1, 1)
            def animate(i):
                ax.clear()
                ax.plot( range(self.num_rings), sltn.iloc[i,:], 'k-', label = "Microseconds: " + str(int(self.t[i])) + f' / {int(self.tfin)}',linewidth = 3.0 )
                ax.axvline( x = p.diffusion['synapse_zone']/self.dr, color='k', ls='--' )
                plt.axvline( x = p.diffusion['transition_zone']/self.dr, color='k', ls='--' )
                ax.legend(loc = 'best')
            ani = FuncAnimation(fig, animate, frames=len(self.t), interval = 1, repeat = False)
        elif mode == 'dynamic2d':
            sol_r = np.zeros([int(2*self.num_rings),int(2*self.num_rings), sltn.shape[0]])
            sol_r[int(self.num_rings), int(self.num_rings), :] = sltn.iloc[:,0]
            for ri in range( 1, self.num_rings ):
                for th in p.theta:
                    z = p.polar2z(ri, th)
                    sol_r[ int(self.num_rings) + round(z.real), int(self.num_rings) + round(z.imag) , : ] = sltn.iloc[:, ri]
            fig, ax = plt.subplots(1, 1)
            def animate(i):
                ax.clear()
                theplot = sol_r[:,:,i]
                ax.imshow(theplot, cmap = 'Greys')
            ani = FuncAnimation(fig, animate, frames=len(sol_r[1,1,:]), interval = 1, repeat = False)
        else:
            print('Incompatible arguments')
        return plt.show()


    def mutual_info(self, locA, locB, draw = True):
        sol = sol_pde.sol_ode(self)
        mi = []
        Tmax = int(len(self.t)/1.5) if self.impulses_number > 1 else int(len(self.t)/1.01)#; print(self.t[Tmax])
        tvec = np.linspace(0,Tmax,101) # Tmax = 5000 usec
        timesT = [int(self.t[i]) for i in tvec.astype(int)]
        for T in tqdm(tvec.astype(int)):
            m = 0
            for t in range(len(self.t)-T-1):
                nA  = sol.iloc[t,:]
                nB  = sol.iloc[t+T,:]
                joint_prob, edges = np.histogramdd([ nA, nB ], bins = self.num_rings, density = True); joint_prob /= joint_prob.sum()
                margA = joint_prob.sum(axis=0)
                margB = joint_prob.sum(axis=1)
                if joint_prob[ locA, locB ] == 0. or margA[locA] == 0. or margB[locB] == 0.:
                    x = 0
                else:
                    x = abs(math.log( joint_prob[ locA, locB ]/margA[locA]/margB[locB] , 2))
                m += round(joint_prob[ locA, locB ] * x, 2)
            mi.append(m)
        if draw:
            plt.plot( tvec, mi, 'k-' )
            plt.show()
        return pd.DataFrame( mi, index = timesT )#tvec


    def plot_mutual_concentration(self, locA, locB, T):
        sltn    = sol_pde.sol_ode(self)
        fig, ax = plt.subplots(1, 1)
        def animate(i):
            ax.clear()
            ax.plot( sltn.iloc[i,:], sltn.iloc[i+T,:], 'ko', alpha = .5, label = "Microseconds: " + str(int(self.t[i])) + f' / {int(self.tfin)}' )
            plt.axvline( x = locA, color='k', ls='--' )
            plt.axhline( y = locB, color='k', ls='--' )
            ax.legend(loc = 'best')
        ani = FuncAnimation(fig, animate, frames = len(self.t)-T, interval = 10, repeat = False)
        return plt.show()


    def testing_function(self, locA, locB):
        def uptk(r):
            if r < self.syn_border:
                out = p.diffusion['U_cleft']
            elif r >= self.syn_border and r <= self.prm_border:
                out = p.diffusion['U_cleft'] + p.f_inter(r*self.dr)*( p.diffusion['U_pm'] - p.diffusion['U_cleft'] )
            else:
                out = p.diffusion['U_pm']
            return out
        the_r = np.arange( 0, int(self.num_rings * self.dr) )
        frr = [ uptk(x) for x in the_r ]
        plt.plot(the_r, frr, 'k-')
        return plt.show()
