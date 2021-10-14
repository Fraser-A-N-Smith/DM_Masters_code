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
import pandas as pd
start = timeit.default_timer()



# Calculates Mstar from Mhalo
def Moster(theory_mhalo, y):
    norm_factor = 0.0351 - 0.0247*(redshift/(redshift+1))
    char_mass = 11.590 + 1.195*(redshift/(redshift+1))
    beta = 1.376 - 0.826*(redshift/(redshift+1))
    gamma = 0.608 + 0.329*(redshift/(redshift+1))
    theory_mstar = np.log10(2 * norm_factor * 10**theory_mhalo * ((10**theory_mhalo/10**char_mass)**(-beta) + (10**theory_mhalo/10**char_mass)**gamma)**-1)
    return theory_mstar - y


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

def Vbulge(R,M):#https://ned.ipac.caltech.edu/level5/Sept16/Sofue/Sofue4.html
    return np.sqrt(G*M/R)

# Total velocity (not including Vgas)
def Vtot(Vdm, Vstar, ML):
    Vtotal = np.sqrt(Vdm**2 + (ML * (Vstar**2)))
    return Vtotal

def Vdm_calc_1(r_input, Mdm):
    Vdm = np.sqrt((G * 10**Mdm ) / r_input) # Equation 1, H. Katz et al. 2017
    return Vdm

def eachgal(halo,gal,const = -3.456,slope = 0.406 ):
    for i in range(len(halo)):
        galaxy = gal[i]
        hal = halo[i]
        #R = np.linspace(0,20)
        #Mbar = [(y-const)/slope for y in R]
        R = slope*galaxy+const 

        Vdm_calc_1(R, hal)


    return

####################################################
#HYperparameters
####################################################


####################################################


def Vstar_calc2(r_list, Mstar_log): # Velocity of the stellar disk if not observed
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

ML = 1.111308962394226

#############################################################

# Values of Mhalo and Mstar
log_Mhalo_theory = 12          #np.arange(10, 14.1, 0.1)
log_Mstar_theory = Moster(log_Mhalo_theory, 0)

# Range of radii to be plotted
radius = np.arange(1, 30.1, 0.1)


# Calculates values of Vstar as an array
Vstar_array = np.array(Vstar_calc(radius, log_Mstar_theory))


# Turns Vdm into an array to be plotted
V_dm_array = np.array(Vdm_calc(radius, log_Mhalo_theory, c_NFW(log_Mhalo_theory)))


# Total velocity values to be plotted
Total_Velocity = np.array(Vtot(V_dm_array, Vstar_array, ML))

print('Mhalo:', log_Mhalo_theory)
print('Mstar', log_Mstar_theory)
print('Rvir:', Rvir_calc(log_Mhalo_theory))
print('c:', c_NFW(log_Mhalo_theory))
print('Mdm:', Mdm_NFW_calc(radius, log_Mhalo_theory, c_NFW(log_Mhalo_theory)))
print('Vdm:', V_dm_array)
print('Vstar:', ML*Vstar_array)
print('Vbulge',Vbulge(radius,log_Mhalo_theory))
# Plotting rotation curve

plt.figure(figsize = (15,15),dpi=120)
plt.plot(radius, V_dm_array, 'k-', label = 'DM Velocity')
plt.plot(radius, np.sqrt(ML)*Vstar_array, 'r-', label = 'Baryonic Velocity')
plt.plot(radius, Total_Velocity, 'b-', label = 'Total velocity')

plt.xlabel("Radius - kpc")
plt.ylabel("Velocity - km/s")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper right", borderaxespad=0.)
plt.title('Rotation curve')
plt.show()




###################################################
AM_HaoFu2020 = pd.read_csv('baldry_Bernardi_SMHM.txt',header=None,delim_whitespace=True)
AM_HaoFu2020_Halos = [x for x in pd.Series(AM_HaoFu2020[0]).values][2:] # remove the top two rows
AM_HaoFu2020_Halos = [float(x) for x in AM_HaoFu2020_Halos] # convert the strings into floats
AM_HaoFu2020_galaxies = [x for x in pd.Series(AM_HaoFu2020[1]).values][2:] # remove the top two rows
AM_HaoFu2020_galaxies = [float(x) for x in AM_HaoFu2020_galaxies] # convert the strings into floats
###################################################

#print(AM_HaoFu2020_galaxies)
#print([10**x for x in AM_HaoFu2020_galaxies])

eachgal(AM_HaoFu2020_Halos,AM_HaoFu2020_galaxies)
##################################################
def kneefunc(xvals,splitpoint=8.6,slope1=1.052,slope2=0.461,constant1=0.236,constant2=5.329):
    #DOI:10.1088/0004-637X/809/2/146 
    #Fig-5 Slope Parameters

    #slope:
    #M∗<10^8.6Mdot slope: 1.052±0.058 constant: 0.236±0.476
    #M*>10^8.6Mdot slope: 0.461±0.011 constant: 5.329±0.112

    #kneefuncx1 = np.linspace(0,8.6,50)
    kneefuncx1 = [x for x in xvals if x<splitpoint]
    kneefuncy1 = [(slope1*x+constant1) for x in kneefuncx1]

    kneefuncx2 = [x for x in xvals if x>= splitpoint]#np.linspace(8.6,14,50)
    kneefuncy2 = [(slope2*x+constant2) for x in kneefuncx2 if x>=splitpoint]

    kneefuncx = np.append([kneefuncx1],kneefuncx2)
    kneefuncy = np.append([kneefuncy1],kneefuncy2) # y is mgas

    Mbar = np.add(kneefuncx,kneefuncy)#Mgal+Mgas
    Mbar = [10**x for x in Mbar] # remove if nescessary
    return Mbar

def Mbarval(xval,splitpoint=8.6,slope1=1.052,slope2=0.461,constant1=0.236,constant2=5.329):
    #DOI:10.1088/0004-637X/809/2/146 
    #Fig-5 Slope Parameters

    #slope:
    #M∗<10^8.6Mdot slope: 1.052±0.058 constant: 0.236±0.476
    #M*>10^8.6Mdot slope: 0.461±0.011 constant: 5.329±0.112

    #kneefuncx1 = np.linspace(0,8.6,50)
    kneefuncx1 = [x for x in xvals if x<splitpoint]
    kneefuncy1 = [(slope1*x+constant1) for x in kneefuncx1]

    kneefuncx2 = [x for x in xvals if x>= splitpoint]#np.linspace(8.6,14,50)
    kneefuncy2 = [(slope2*x+constant2) for x in kneefuncx2 if x>=splitpoint]

    kneefuncx = np.append([kneefuncx1],kneefuncx2)
    kneefuncy = np.append([kneefuncy1],kneefuncy2) # y is mgas

    Mbar = np.add(kneefuncx,kneefuncy)#Mgal+Mgas
    Mbarval = [10**x for x in Mbar] # remove if nescessary

    return Mbarval

Mbar = kneefunc(AM_HaoFu2020_galaxies)

##################################################
eachgal(AM_HaoFu2020_Halos,AM_HaoFu2020_galaxies)
##################################################
#
#################################################
#DOI:10.1088/0004-637X/809/2/146 
#Fig-5 Slope Parameters

slope = 0.406
const = -3.456 

#xvals = [10**x for x in Mbar]#
xvals =np.logspace(0.0,12.0)
yvals = [np.log10(x)*slope+const for x in xvals]
xvals = [np.log10(x) for x in xvals] # log
plt.plot(xvals,yvals,label=("theory"))

#TRIPLE LOGGED VALUES FIX THIS ASAP
#

xvals = Mbar
yvals = [np.log10(x)*slope+const for x in xvals] # log
xvals = [np.log10(x) for x in xvals]
plt.plot(xvals,yvals,label=("our range"))
#plt.xlim(7.,12.)
#plt.ylim(-1.,1.5)
plt.xlabel("M$_{Baryon}$[M$_{\odot}$]")
plt.ylabel("log(R$_{reff}$)[kpc]")
plt.legend()
plt.show()

log_Reff = yvals
#print(log_Reff)
Rd = [(x/1.68) for x in log_Reff] #unlog
R = [x*2.2 for x in Rd]
#print(R)
#print(AM_HaoFu2020_Halos)
#Vdm = [Vdm_calc(10**x) for x in ]
plt.plot(R,xvals)
plt.xlabel('R')
plt.ylabel('Mdm')
plt.show()
Vdm = []
for i in range(len(R)):
    Vdm.append(Vdm_calc_1(np.log10(AM_HaoFu2020_Halos[i]),R[i])) # order of 0.001 kpc # wrong calc, dm halo assumed whole mass within
    #radius R.


slope = 0.406
const = -3.456 

#xvals = [10**x for x in Mbar]#
xvals =np.logspace(0.0,12.0)
yvals = [np.log10(x)*slope+const for x in xvals]
xvals = [np.log10(x) for x in xvals] # log
plt.plot(xvals,yvals,label=("theory"))

#TRIPLE LOGGED VALUES FIX THIS ASAP
#

xvals = Mbar
yvals = [np.log10(x)*slope+const for x in xvals] # log
xvals = [np.log10(x) for x in xvals]
plt.plot(xvals,yvals,label=("our range"))
#plt.xlim(7.,12.)
#plt.ylim(-1.,1.5)
plt.xlabel("M$_{Baryon}$[M$_{\odot}$]")
plt.ylabel("log(R$_{reff}$)[kpc]")
plt.legend()
plt.show()

log_Reff = yvals
#print(log_Reff)
Rd = [(x/1.68) for x in log_Reff] #unlog
R = [x*2.2 for x in Rd]

def Mbar_radius(Radius,slope = 0.406,const = -3.456 ):
    Mbar = (Radius-const)/slope
    Mbar = np.append([Mbar],Mbar)

    return Mbar



#print(R,Vdm)

plt.plot(R,Vdm)
plt.xlabel('R')
plt.ylabel('Vdm')
plt.show()
#Vcirc = np.sqrt(Vdm^2+Vgal^2)

##################################################
Vbar = np.array(Vstar_calc(radius, Mbar_radius(radius[0])))#logMbar))

print('Mhalo:', log_Mhalo_theory)
print('Mstar', log_Mstar_theory)
print('Rvir:', Rvir_calc(log_Mhalo_theory))
print('c:', c_NFW(log_Mhalo_theory))
print('Mdm:', Mdm_NFW_calc(radius, log_Mhalo_theory, c_NFW(log_Mhalo_theory)))
print('Vdm:', V_dm_array)
print('Vstar:', ML*Vstar_array)
print('Vbulge',Vbulge(radius,log_Mhalo_theory))
# Plotting rotation curve
plt.figure(figsize = (15,15),dpi=120)
plt.plot(radius, V_dm_array, 'k-', label = 'DM Velocity')
plt.plot(radius,Vbar,label='Baryonic Velocity')
plt.plot(radius, Total_Velocity, 'b-', label = 'Total velocity')
plt.xlabel("Radius - kpc")
plt.ylabel("Velocity - km/s")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper right", borderaxespad=0.)
plt.title('Rotation curve')
plt.show()

##################################################
def plotHaoData(Galaxies,Halos):
    error = 0.3
    sigma2 = error*2
    AM_HaoFu2020_galaxies= np.asarray(Galaxies)
    AM_HaoFu2020_Halos = np.asarray(Halos)
    plt.plot(AM_HaoFu2020_Halos,AM_HaoFu2020_galaxies)
    plt.title('SPARC DATASET - AM - '+model)
    plt.ylabel('$log_{10}[M_{*}/M_{\odot}]$')
    plt.fill_between(AM_HaoFu2020_Halos,AM_HaoFu2020_galaxies+error,AM_HaoFu2020_galaxies-error,alpha=0.5)
    plt.fill_between(AM_HaoFu2020_Halos,AM_HaoFu2020_galaxies+sigma2,AM_HaoFu2020_galaxies-sigma2,alpha=0.5)
    plt.xlabel('$log_{10}[M_{Halo}/M_{\odot}]$')
    plt.savefig('SPARC DATASET-AM-'+model+"_error_"+'point2dex'+'_bounded')
    plt.show()
    return

plotHaoData(AM_HaoFu2020_galaxies,AM_HaoFu2020_Halos)
#print(AM_HaoFu2020_galaxies,AM_HaoFu2020_Halos)


