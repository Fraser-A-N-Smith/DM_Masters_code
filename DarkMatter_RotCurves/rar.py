#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 10:05:15 2021

@author: fraser
"""

#Import nescessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import exp
import math
from scipy.special import kv, iv
import matplotlib.font_manager as fm
from scipy.interpolate import interp1d
import scipy.optimize as op
from scipy.integrate import quad
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from scipy.integrate import cumtrapz
from scipy.integrate import trapz
from scipy.special import erfc

#SET HYPERPARAMETERS ###################

G = 6.67e-11
A = 50.0 #Mdot k^-4 s^4 


########################################
#Define Functions

def theory_tf(x, m, b):
    return ((x - b + m) / m)

def theory_g(x, g_scale):    #function fitting char(), McGaugh 2016
    return x / (1 - np.exp(- np.sqrt(x/g_scale))) 

def acc_bar(r_input,ML):
    V_bar = np.sqrt(np.sign(Vgas(galaxy_name))*(Vgas(galaxy_name)**2)+(ML*(Vstar(galaxy_name)**2))+(ML*1.4*(Vbulge(galaxy_name)**2)))
    g = ((V_bar**2) / r_input) / 3.086e13 
    return g
#########################################

#from Hopkins paper g_obs = sqrt(g_bar x g_daggeer) g_dagger = (G x FancyA)^-1 FancyA = 50 Modot k^-4 s^4    Mgaugh 200

g_bar = np.logspace(-12,-9,100)
g_obs = [np.sqrt(x*(G*A)**(-1.0)) for x in g_bar]

log_g_bar = [np.log10(x) for x in g_bar]
log_g_obs = [np.log10(x) for x in g_obs]


plt.plot(log_g_bar,log_g_obs)

plt.show()

########################################
alpha = 3.75
alpha_error = 0.11
beta = 9.5
beta_error = 0.013
pivot = 1.915
theory_m = np.linspace(5.3, 11.5)
theory_v = ((theory_m - beta + (alpha * pivot)) / alpha)
g = 9.81
plt.plot(theory_m, theory_v, color = "black", linestyle='--')
plt.show()

#######################################

popt, pcov = op.curve_fit(theory_g, g_bar, g, p0 = 1.2e-10) 
theory_1to1 = np.linspace(10**-12.5, 10**-9)
plt.plot(theory_1to1, theory_g(theory_1to1, *popt))     #Our best fit
plt.plot(theory_1to1, theory_1to1, linestyle='--') #No DM, 1:1 line
plt.plot(theory_1to1, theory_g(theory_1to1, 1.2e-10))    #McGaugh fit 
plt.xlabel("$g_{bar} [m s^{-2}]$")
plt.ylabel("$g [m s^{-2}]$")
#plt.xlim(10**-12.5, 10**-9)
#plt.ylim(10**-12, 10**-9)
plt.xscale('log')
plt.yscale('log')
plt.show()