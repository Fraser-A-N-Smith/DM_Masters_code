#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 16:01:05 2021

@author: fraser
"""

#############################IMPORT MODULES
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy.optimize as op
from scipy import interpolate
import corner
import emcee
from scipy.special import kv, iv
from scipy.integrate import quad
import timeit
import random
import seaborn as sns
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from scipy.integrate import trapz
from scipy.special import erfc

cosmo = cosmology.setCosmology("planck18")
Cosmo = cosmology.getCurrent()
plt.rcParams['font.size'] = 18 # changes font size
plt.rcParams['axes.linewidth'] = 2 # changes linewidth
#############################


###############################HYPERPARAMETERS
z = 0
redshift = z
###############################

#-----------------------------------------------------------------#

############################
#Import Functions

#Import Moster 
def Moster(theory_mhalo, y):
    norm_factor = 0.0351 - 0.0247*(redshift/(redshift+1))
    char_mass = 11.590 + 1.195*(redshift/(redshift+1))
    beta = 1.376 - 0.826*(redshift/(redshift+1))
    gamma = 0.608 + 0.329*(redshift/(redshift+1))
    theory_mstar = np.log10(2 * norm_factor * 10**theory_mhalo * ((10**theory_mhalo/10**char_mass)**(-beta) + (10**theory_mhalo/10**char_mass)**gamma)**-1)
    return theory_mstar - y

#Explanation: The moster function takes halo mass values and a -ive y offset as arguments
#it then determines the stellar mass per the halo masses given as arguments 

def Moster(theory_mhalo, y):
    norm_factor = 0.0351 - 0.0247*(redshift/(redshift+1))
    char_mass = 11.590 + 1.195*(redshift/(redshift+1))
    beta = 1.376 - 0.826*(redshift/(redshift+1))
    gamma = 0.608 + 0.329*(redshift/(redshift+1))
    theory_mstar = np.log10(2 * norm_factor * 10**theory_mhalo * ((10**theory_mhalo/10**char_mass)**(-beta) + (10**theory_mhalo/10**char_mass)**gamma)**-1)
    return theory_mstar - y

def Mn(z,Mn01,Mnz):
    Mn = Mn01+Mnz*((z-0.1)/(z+1))
    return Mn
    
def N(z,N01,Nz):
    return N01 + Nz*((z-0.1)/(z+1))
    
def Beta(z,B01,Bz):
    return B01 + Bz*((z-0.1)/(z+1))
    
def Gamma(z,G01,Gz):
    return G01 + Gz*((z-0.1)/(z+1))

# Mn01,Mnz,N01,Nz,B01,Bz,G01,Gz
# 11.91,0.52,0.029,-0.018,2.09,-1.03,0.64,0.084
# 11.91,0.58,0.032,-0.014,1.64,-0.69,0.53,0.03
def PGrylls19(Mhalo,z,Mn01,Mnz,N01,Nz,B01,Bz,G01,Gz):
    
    Mstar = 2.0*Mhalo*N(z,N01,Nz)*((Mhalo/Mn(z,Mn01,Mnz))**(-Beta(z,B01,Bz))+(Mhalo/Mn(z,Mn01,Mnz))**(Gamma(z,G01,Gz)))**(-1.0)
    
    return Mstar



def integrals(SMF , HMF , sig):

    #Returns the integrals of the Stellar Mass Function (SMF) and HMF as calculated in Aversa et al. (2015) eq. 37

    M_s , phi_s = SMF[0] , SMF[1]
    M_h , phi_h = HMF[0] , HMF[1]
    
    phi_s , phi_h = 10.0**phi_s , 10.0**phi_h
    #phi_s = [10.0**x for x in phi_s]
    #phi_h = [10.0**x for x in phi_h]
    
    I_phi_s = np.flip(cumtrapz(np.flip(phi_s), M_s))

    I_phi_h = np.array([])
    for m in M_h:
        I = np.trapz(phi_h*0.5*erfc((m-M_h)/(np.sqrt(2)*sig)) , M_h)
        I_phi_h = np.append(I_phi_h,I)

    M_s , M_h = M_s+0.025 , M_h+0.025 # original code read M_s , M_h = M_s[:-1]+0.025 , M_h+0.025

    return I_phi_s , I_phi_h , M_s , M_h

def SMFfromSMHM(M_s , M_h , sig_0 , z):

    #Reconstructs the SMF from the Stellar Mass Halo Mass (SMHM) relationship

    bins = 0.1
    volume = (500*cosmo.h)**3

    M_hmf , phi = HMF(z , bins , cum_var = volume*bins)
    M_hmf = M_hmf[20:] #Cutting the low mass end of the HMF for increased speed
    phi = 10**phi[20:]

    cum_phi = np.cumsum(phi)
    max_number = np.floor(np.max(cum_phi))
    if (np.random.uniform(0,1) > np.max(cum_phi)-max_number): #Calculating number of halos to compute
        max_number += 1

    int_cum_phi = interp1d(cum_phi, M_hmf)
    range_numbers = np.random.uniform(np.min(cum_phi), np.max(cum_phi), int(max_number))
    halo_masses = int_cum_phi(range_numbers)

    M_smf = np.arange(8.5, 12.2 , 0.1) #SMF histogram bins
    SMHMinterp = interp1d(M_h , M_s , fill_value="extrapolate")
    stellar_masses = SMHMinterp(halo_masses) + np.random.normal(0., sig_0, halo_masses.size) #Calculating stellar masses using SMHM with scatter
    phi_smf = np.histogram(stellar_masses , bins = M_smf)[0]/0.1/volume #SMF histogram

    return M_smf[:-1]+0.05 , np.log10(phi_smf)

def HMF(z , bins = 0.05 ,cum_var = 1.0):

    #Produces a Halo Mass Function (HMF) using a Tinker et al. (2008) HMF and a correction for subhalos from Behroozi et al. (2013)

    M = np.arange(8. , 15.5+bins ,bins) #Sets HMF mass range
    phi = mass_function.massFunction((10.0**M)*cosmo.h, z , mdef = '200m' , model = 'tinker08' , q_out="dndlnM") * np.log(10) * cosmo.h**3.0 * cum_var #Produces the Tinker et al. (2008) HMF

    a = 1./(1.+z) #Constants for Behroozi et al. (2013) Appendix G, eqs. G6, G7 and G8 subhalos correction
    C = np.power(10., -2.415 + 11.68*a - 28.88*a**2 + 29.33*a**3 - 10.56*a**4)
    logMcutoff = 10.94 + 8.34*a - 0.36*a**2 - 5.08*a**3 + 0.75*a**4
    correction = phi * C * (logMcutoff - M)

    return M , np.log10(phi + correction)



###########################

#------------------------------------------------------------------#

########################### GENERATE DATA

#import Hao Fu Data
AM_HaoFu2020 = pd.read_csv('baldry_Bernardi_SMHM.txt',header=None,delim_whitespace=True)
AM_HaoFu2020_Halos = [x for x in pd.Series(AM_HaoFu2020[0]).values][2:] # remove the top two rows
AM_HaoFu2020_Halos = [float(x) for x in AM_HaoFu2020_Halos] # convert the strings into floats
AM_HaoFu2020_galaxies = [x for x in pd.Series(AM_HaoFu2020[1]).values][2:] # remove the top two rows
AM_HaoFu2020_galaxies = [float(x) for x in AM_HaoFu2020_galaxies] # convert the strings into floats


#import Halo Mass Function
HaloMassFunc = pd.read_csv("HaloMassFunction.txt",header=None,delim_whitespace=True)
HMF_M = [x for x in pd.Series(HaloMassFunc[0]).values]
HMF_P = [x for x in pd.Series(HaloMassFunc[1]).values]
Haloforintegration = [[],[]]
Haloforintegration[0] = HMF_M
Haloforintegration[1] = HMF_P

#generate halo masses
halomasses = np.log10(np.logspace(9.5,14))


#create HMF
ShankarHalo = pd.read_csv('shankarhalos2021.csv',header=None,delim_whitespace=True)
ShankarHalo_M = [float(x) for x in pd.Series(ShankarHalo[0]).values]
ShankarHalo_P = [float(x) for x in pd.Series(ShankarHalo[1]).values]


#generate stellar masses via moster
mosterstellarmasses = [Moster(x,0) for x in halomasses]

#scatter moster SMF


#import scattered halo data
point2leja = pd.read_csv('(FR) Leja SMHM sigma = 0.2.txt',header=None,delim_whitespace=True,skiprows=2)
point2lejaMh = [x for x in pd.Series(point2leja[0]).values]
point2lejaGl = [x for x in pd.Series(point2leja[1]).values]
point4bernardi = pd.read_csv('(FR) Baldry SMHM sigma = 0.4.txt',header=None,delim_whitespace=True,skiprows=2)
point4logMh = [x for x in pd.Series(point4bernardi[0]).values]
point4logGl = [x for x in pd.Series(point4bernardi[1]).values]
point2bernardi = pd.read_csv('(FR) Baldry SMHM sigma = 0.2.txt',header=None,delim_whitespace=True,skiprows=2)
point2logMh = [x for x in pd.Series(point2bernardi[0]).values]
point2logGl = [x for x in pd.Series(point2bernardi[1]).values]
point6bernardi = pd.read_csv('(FR) Baldry SMHM sigma = 0.6.txt',header=None,delim_whitespace=True,skiprows=2)
point6logMh = [x for x in pd.Series(point6bernardi[0]).values]
point6logGl = [x for x in pd.Series(point6bernardi[1]).values]

#generate galaxy masses for halo masses
Mgalaxies = [PGrylls19(Mhalo, 0.1, 11.91,0.52,0.029,-0.018,2.09,-1.03,0.64,0.084) for Mhalo in halomasses]

#print(np.log10(Mgalaxies),halomasses)

###########################

#-------------------------------------------------------------------#

###########################PLOT DATA

plt.figure(dpi=60,figsize=(12,12))
plt.plot(point2lejaMh, point2lejaGl, label="Leja, $\sigma$:0.2" )
plt.plot(point4logMh, point4logGl,label="Bernardi, $\sigma$:0.4")
plt.plot(point2logMh, point2logGl,label="Bernardi, $\sigma$:0.2")
plt.plot(point6logMh, point6logGl,label="Bernardi, $\sigma$:0.6")
#plt.plot(halomasses,np.log10(Mgalaxies),label="P.Grylls 2019")
plt.xlabel('log(M$_{Halo}$/M$_\odot$)')
plt.ylabel('log(M$_{Gal}$/M$_\odot$)')
plt.legend()
plt.show()



#generate figure of Mstar vs Mhalo
plt.figure(figsize=(12,12),dpi=120)
plt.ylabel('log(M$_{*}$/M$_\odot$)',fontname='DejaVu Sans')
plt.xlabel("log(M$_{Halo}$/M$_\odot$)")
plt.xlim(7,14)
plt.plot(halomasses,mosterstellarmasses,label='Moster')
plt.plot(AM_HaoFu2020_Halos,AM_HaoFu2020_galaxies,label='Bernardi-Baldry')
plt.show()

###########################