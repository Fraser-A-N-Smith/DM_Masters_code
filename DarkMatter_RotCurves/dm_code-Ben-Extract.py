# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:20:17 2017

Python 3.6

@author: Nic
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

#r_list = np.array([0.72,1.43,2.16,2.87,3.59,4.3,5.03,5.75,6.46,7.18,7.91,8.62,11.49,14.3,17.19,20.07,22.96,25.85,28.74,31.62,34.51,37.4,40.29,43.17,45.92,48.81,51.7,54.59]) # The radii data
#Vtot_data = np.array([125.0,162.0,178.0,193.0,192.0,190.0,195.0,201.0,204.0,204.0,206.0,206.0,206.0,206.0,203.0,200.0,194.0,188.0,184.0,182.0,180.0,181.0,180.0,179.0,179.0,179.0,174.0,172.0])# The total velocity curve data
#Verr = np.array([17.6,8.5,5.22,1.88,3.59,0.97,0.99,0.45,0.81,0.65,0.66,1.54,1.9,0.42,4.19,1.23,4.09,5.2,1.16,2.47,3.85,8.43,4.55,0.44,1.96,2.99,3.39,4.84])
#Vstar = np.array([201.48,240.13,263.27,280.74,273.32,272.18,273.73,276.2,279.72,275.62,271.43,267.09,245.58,225.03,208.52,190.41,175.86,164.23,154.38,146.2,139.3,133.34,128.12,123.52,119.59,115.83,112.43,109.32])
#Vgas = np.array([2.2,3.72,4.09,5.34,6.03,5.11,8.42,14.42,19.11,22.51,25.8,28.49,33.07,45.73,47.26,44.56,40.45,37.41,38.46,40.22,42.57,40.57,41.86,42.92,44.68,44.56,43.74,41.74])
"Select input variables"

# Testing using D564-8, file name "DDO64" in SPARC folder

"Choose model; NFW or DC14 or WDM"
# Write the name of the model here. Ensure that it is written exactly as in the list
model = "DC14"
redshift = 0.0 #currently for 0.0 or 1.1
"""

'NFW' - NFW
'DC14' - Di Cinto et al. 2014
'WDM1' - Warm Dark Matter, thermal with m_X = 3keV
'WDM2' - Warm Dark Matter, m_nu = 7keV, sin^2(2theta)=210^-10
'WDM3' - Warm Dark Matter, m_nu = 7keV, sin^2(2theta)=510^-11
"""

"Limits for optimal values of Mhalo_log, concentration and Mass-to-Light ratio"
"Variable_lim = np.array([lower bound,upper bound])"

Mhalo_log_lim = np.array([8, 14])
c_lim= np.array([1, 100])
ML_lim = np.array([0.02, 5.0])

"Are the stellar and gas velocities provided?"
Vprovided = True # If Vstar and Vgas are given from data then set as 'True'. If not, set as 'False'

wdm_dist = 0
halo_dist = 0
g_ML = 0
g_Mhalo = 0
Rd = 1
al_gas = 1
Ms = 5
# Ms is the input free-streaming  scale for the WDM model

h = 0.7

galaxy_name = "NGC5055"
#galaxy_name = 'KK98-251'

under30 = ['CamB', 'D564-8', 'PGC51017', 'UGC04483', 'UGCA281']
under40 = ['CamB', 'D512-2', 'D564-8', 'F574-2', 'KK98-251', 'PGC51017', 'UGC04483', 'UGCA281', 'UGCA444']
"Reads data from the SPARC file"

#==============================================================================
# Importing Data
#==============================================================================
f = open("rotationCurvesSPARC.txt", "r")
data_lines = f.readlines()
f.close()
f = open("galaxymassesSPARC.txt", "r")
data_lines1 = f.readlines()
f.close()
f = open("centralLumSPARC.txt", "r")
data_lines2 = f.readlines()
f.close()

def r_list(galaxy):
    r_list=np.array([])
    for i in range(len(data_lines)):
        temp2 = data_lines[i]
        temp2 = temp2.split()
        if str(temp2[0]) == (galaxy):
            r_list = np.append([r_list],float(temp2[2]))
    return r_list

def Vtot_data(galaxy):
    Vtot_data=np.array([])
    for i in range(len(data_lines)):
        temp2 = data_lines[i]
        temp2 = temp2.split()
        if str(temp2[0]) == (galaxy):
            Vtot_data = np.append([Vtot_data],float(temp2[3]))
    return Vtot_data

def Verr(galaxy):
    Verr=np.array([])
    for i in range(len(data_lines)):
        temp2 = data_lines[i]
        temp2 = temp2.split()
        if str(temp2[0]) == (galaxy):
            Verr = np.append([Verr],float(temp2[4]))
    return Verr

def Vgas(galaxy):
    Vgas=np.array([])
    for i in range(len(data_lines)):
        temp2 = data_lines[i]
        temp2 = temp2.split()
        if str(temp2[0]) == (galaxy):
            Vgas = np.append([Vgas],float(temp2[5]))
    return Vgas

def Vstar(galaxy):
    Vstar=np.array([])
    for i in range(len(data_lines)):
        temp2 = data_lines[i]
        temp2 = temp2.split()
        if str(temp2[0]) == (galaxy):
            Vstar = np.append([Vstar],float(temp2[6]))
    return Vstar
    
def Vbulge(galaxy):
    Vbulge=np.array([])
    for i in range(len(data_lines)):
        temp2 = data_lines[i]
        temp2 = temp2.split()
        if str(temp2[0]) == (galaxy):
            Vbulge = np.append([Vbulge],float(temp2[7]))
    return Vbulge

def Bright(galaxy):
    for i in range(len(data_lines2)):
        temp2 = data_lines2[i]
        temp2 = temp2.split()
        if str(temp2[0]) == (galaxy):
            centralLum = np.log10(float(temp2[12]))
    return centralLum

def Lum(galaxy):
    for i in range(len(data_lines2)):
        temp2 = data_lines2[i]
        temp2 = temp2.split()
        if str(temp2[0]) == (galaxy):
            lum = float(temp2[7])
    return lum

def e_Lum(galaxy):
    for i in range(len(data_lines2)):
        temp2 = data_lines2[i]
        temp2 = temp2.split()
        if str(temp2[0]) == (galaxy):
            lum = float(temp2[8])
    return lum
    
def TF_V(galaxy):
    for i in range(len(data_lines1)):
        temp2 = data_lines1[i]
        temp2 = temp2.split()
        if str(temp2[0]) == (galaxy):
            TF_V = float(temp2[6])
    return TF_V

def TF_M(galaxy):
    for i in range(len(data_lines1)):
        temp2 = data_lines1[i]
        temp2 = temp2.split()
        if str(temp2[0]) == (galaxy):
            TF_M = float(temp2[4])
    return TF_M

def H1Mass(galaxy):
    for i in range(len(data_lines2)):
        temp2 = data_lines2[i]
        temp2 = temp2.split()
        if str(temp2[0]) == (galaxy):
            H1 = float(temp2[13])
    return H1
galaxy_names = np.array([])        
for i in range(len(data_lines1)):
    temp2 = data_lines1[i]
    temp2 = temp2.split()
    galaxy_names = np.append(galaxy_names, temp2[0])
galaxy_names = np.unique(galaxy_names)
#print (len(galaxy_names))
f = open('tab_3kev.dat',"r") 
c_data1 = f.readlines()
f.close()
f = open('tab_RP_7keV_2e-10.dat',"r")
c_data2 = f.readlines()
f.close()
f = open('tab_RP_7keV_5e-11.dat',"r")
c_data3 = f.readlines()
f.close()
