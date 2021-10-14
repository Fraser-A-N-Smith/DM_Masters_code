# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 12:31:57 2020

@author: brad2
"""

from __future__ import print_function
import math
import scipy.optimize as op
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
import corner
import emcee
from scipy.special import kv, iv
from scipy.integrate import quad
import timeit
start = timeit.default_timer()



# Calculates Mstar from Mhalo
def Moster(theory_mhalo, y):
    norm_factor = 0.0351 - 0.0247*(redshift/(redshift+1))
    char_mass = 11.590 + 1.195*(redshift/(redshift+1))
    beta = 1.376 - 0.826*(redshift/(redshift+1))
    gamma = 0.608 + 0.329*(redshift/(redshift+1))
    theory_mstar = np.log10(2 * norm_factor * 10**theory_mhalo * ((10**theory_mhalo/10**char_mass)**(-beta) + (10**theory_mhalo/10**char_mass)**gamma)**-1)
    return theory_mstar - y

# Values of Mhalo and Mstar
log_Mhalo_theory = 12          #np.arange(10, 14.1, 0.1)
log_Mstar_theory = Moster(log_Mhalo_theory, 0)

# Range of radii to be plotted
radius = np.arange(1, 30.1, 0.1)

'''
Functions used to calculate values
'''
def Rvir_calc(Mhalo_log):   #same as R200
    z = redshift # Line 241 from ForAnaCamila
    omega_m = 0.3 # Line 242 from ForAnaCamila
    HH = 0.1 * h * math.sqrt((omega_m * ((1+z)**3)) + (1-omega_m)) # Line 410,411 from ForAnaCamila
    rho_c = (3 * (HH**2)) / (8 * math.pi * G) # Line 413 from ForAnaCamila
    k = (4*math.pi) / 3 # Line 414 from ForAnaCamila
    Rvir_result = np.cbrt(((10 ** Mhalo_log) / (rho_c * 200 * k))) # Check order of operations
    return Rvir_result

# Concentration in NFW model
def c_NFW(Mhalo_log):
    alpha = 10.84
    gamma = 0.085
    M_param = 5.5e17
    theory_mhalo_units = 10**Mhalo_log*1e-12*h
    c = 10**(np.log10((alpha *(1/theory_mhalo_units)**gamma)*(1+(theory_mhalo_units/(M_param*1e-12))**0.4))+wdm_dist)
    return c

# Mass of DM in NFW
def Mdm_NFW_calc(r_input, Mhalo_log, c):# Calculation for dark matter halo mass under the NFW model
    r = r_input
    Rvir = Rvir_calc(Mhalo_log)
    Rs = Rvir / c
    x = r / Rs # Line 488 From ForAnaCamila. Check whether r is under logarithm

    gx = np.log(1+x) -  (x / (1 + x))
    gc = np.log(1+c) - (c / (1 + c))
    Mdm_result = Mhalo_log + np.log10(gx) - np.log10(gc) # Line 490 from ForAnaCamila
    return Mdm_result

# Velocity of DM
def Vdm_calc(r_input, Mhalo_log, c):
    Mdm = Mdm_NFW_calc(r_input, Mhalo_log, c)
    Vdm = np.sqrt((G * 10**Mdm ) / r_input) # Equation 1, H. Katz et al. 2017
    return Vdm


# Turns Vdm into an array to be plotted
V_dm_array = np.array(Vdm_calc(radius, log_Mhalo_theory, c_NFW(log_Mhalo_theory)))

def Vstar_calc(r_list, Mstar_log): # Velocity of the stellar disk if not observed
    Vstar = np.array([])
    for i in range(len(r_list)):
        r = r_list[i]
        a = r / Rd # Using a as a temporary variable
        a_s = a * 0.5 # a_s is half of a
        B = ( iv(a_s, 1e-15) * kv(a_s,1e-15) ) - ( iv(a_s,1) * kv(a_s,1) ) # iv and kv are modified Bessel functions
        V = math.sqrt( (0.5 * G * a * a * B * (10 ** Mstar_log)) / Rd) # Unsure about the total function in line 455 of ForAnaCamila
        a = np.append(Vstar,[V])
        Vstar = a
    return Vstar

####################################################
#HYperparameters
####################################################

"Choose model; NFW or DC14 or WDM"
# Write the name of the model here. Ensure that it is written exactly as in the list
model = "NFW"
redshift = 0.0 #currently for 0.0 or 1.1

wdm_dist = 0
halo_dist = 0
g_ML = 0
g_Mhalo = 0
Rd = 1
al_gas = 1
Ms = 5
# Ms is the input free-streaming  scale for the WDM model

h = 0.7

G = 4.302*(10**(-6)) # The gravitational constant in Kpc/Msun(km/s)^2

ML = 0.5

###################################################


# Calculates values of Vstar as an array
Vstar_array = np.array(Vstar_calc(radius, log_Mstar_theory))

# Total velocity (not including Vgas)
def Vtot(Vdm, Vstar, ML):
    Vtotal = np.sqrt(Vdm**2 + (ML * (Vstar**2)))
    return Vtotal

# Total velocity values to be plotted
Total_Velocity = np.array(Vtot(V_dm_array, Vstar_array, ML))

print('Mhalo:', log_Mhalo_theory)
print('Mstar', log_Mstar_theory)
print('Rvir:', Rvir_calc(log_Mhalo_theory))
print('c:', c_NFW(log_Mhalo_theory))
print('Mdm:', Mdm_NFW_calc(radius, log_Mhalo_theory, c_NFW(log_Mhalo_theory)))
print('Vdm:', V_dm_array)
print('Vstar:', ML*Vstar_array)

# Plotting rotation curve
plt.plot(radius, V_dm_array, 'k-', label = 'DM Velocity')
plt.plot(radius, np.sqrt(ML)*Vstar_array, 'r-', label = 'Stellar Velocity')
plt.plot(radius, Total_Velocity, 'b-', label = 'Total velocity')
plt.xlabel("Radius - kpc")
plt.ylabel("Velocity - km/s")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Rotation curve')
plt.show()