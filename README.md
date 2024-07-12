The main routine for calculations is `LAUNCH.py`, which contains commands to solve the PDE and use this solution for mutual information (MI) calculations.

The solutions are stored in a separate folder as CSV files. The functions for MI take these files according to the desired number of impulses and clearance rate, and place the result of the calculations in a different folder.

The modules folder contains routines for parameter storage (`pars_synapse.py`), the main routine with the PDE solution (`pde_synapse.py`), and the MI calculations (`miProcessing.py`). 
