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
import random
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
model = input('enter model, NFW DC14 or WDM: ')
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

galaxy_name = input("galaxy_name: ")
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
"""
This code has been created using Francesco's guidance, a previous version in IDL and relevant papers.
Below I have given a summary of what the code does and for anyone making improvements of it, the current
issues that I have been tackling.

WHAT THE CODE DOES:

Firstly, the code has multiple functions for each model that predict the value of M(<r) (Halo mass contained
in radius r). The model may be selected by typing its name below this large section of commenting where it
says model = " (type model here) "

Secondly, using this calculated value of M(<r) the total velocity curve is calculated from the two equations:
    
Vdm = sqrt( ( G * M(<r) ) / r)

Vtotal = sqrt(Vdm**2 + Vgas**2 + (ML * Vstar**ML))

And Vtotal is plotted against r, giving the velocity curve plot.

Finally, this large calculation is passed through the optimize.curve_fit function, which finds optimal
values of logMhalo, concentration and ML (Mass-to-light ratio) to match the calculation to the observed
total velocity which is given in the data sets.

The suitability of the model may then be discussed from whether these optimal values lie within expectations.

The SPARC data set is available here: http://astroweb.cwru.edu/SPARC/

INSTRUCTIONS FOR ALTERING GAUSSIAN DISTRIBUTIONS:
    
To include/remove the gaussian distribution for the Mhalo-c relation for the NFW and DC14 models as well as the WDM
models the code on lines 371 and 469 must be changed according to the instructions on those lines (Vdm_calc
and opline functions)

To include/remove the gaussian distribution for the Mhalo-Mstar relation the code on lines 388 and 472 must be
changed according to the instructions on those lines (Vtot_calc and opline functions)
"""
# If any inputs are unknown then leave as 0
# Mstar_log: Stellar mass of galaxy
# Mgas_log: Mass of gas
# Mhalo_log: Mass of halo
# Rd: Disk scale length in kpc
# ML: Mass-to-light ratio
# Vstar is the stellar disk velocity, if known
# Vgas is the gas disk velocity, if known
# al_gas is a coefficient for the assumed relationship between the scale-lengths of the
    # stellar and gas disks. It is generally between 1 and 3
# M prefix represents a mass, R prefix represents a radius
# _log suffix will be used to denote when values are the logarithm of base 10

G = 4.302*(10**(-6)) # The gravitational constant in Kpc/Msun(km/s)^2

# Current inputs / unknowns : Vgas , Vstar , ML , r , c , Mh , z , omega_m , R , h

#==============================================================================
# Defining subfunctions
#==============================================================================

def Rvir_calc(Mhalo_log):   #same as R200
    z = redshift # Line 241 from ForAnaCamila
    omega_m = 0.3 # Line 242 from ForAnaCamila
    HH = 0.1 * h * math.sqrt((omega_m * ((1+z)**3)) + (1-omega_m)) # Line 410,411 from ForAnaCamila
    rho_c = (3 * (HH**2)) / (8 * math.pi * G) # Line 413 from ForAnaCamila
    k = (4*math.pi) / 3 # Line 414 from ForAnaCamila
    Rvir_result = np.cbrt(((10 ** Mhalo_log) / (rho_c * 200 * k))) # Check order of operations
    return Rvir_result

def c_calc(c_file, Mhalo_log):
    #f = open(c_file,"r") #File name of concentration data
    #c_data = f.readlines() # Inputs each line as a section of an array
    c200c = []
    M200c = []
    for n in range(len(c_file)):
        temp1 = c_file[n].split()
#        if (temp1[0:5]) == " 0.00": # If the redshift is zero then c200c and M200c are added to their arrays
        c200c.append(float(temp1[3]))
        M200c.append(np.log10(float(temp1[0])*1e12)) #!!!not sure if log10 should be used here!!!
    MtoC = interpolate.interp1d(M200c,c200c, fill_value = "extrapolate") # Declaring the Mass-to-Concentration interpolation
    # Approximates the value of c at our galaxy mass via interpolation of known data
    interpolated_c = MtoC(Mhalo_log)
    c_final = 10 ** (np.log10(interpolated_c) + wdm_dist)
    #f.close()
    return c_final

def Moster(theory_mhalo, y):
    norm_factor = 0.0351 - 0.0247*(redshift/(redshift+1))
    char_mass = 11.590 + 1.195*(redshift/(redshift+1))
    beta = 1.376 - 0.826*(redshift/(redshift+1))
    gamma = 0.608 + 0.329*(redshift/(redshift+1))
    theory_mstar = np.log10(2 * norm_factor * 10**theory_mhalo * ((10**theory_mhalo/10**char_mass)**(-beta) + (10**theory_mhalo/10**char_mass)**gamma)**-1)
    return theory_mstar - y

def halo_calc(ML):
    Mh = [10,10.25,10.5,10.75,11,11.25]
    Ms = [7.5,7.8,8.1,8.5,9.1,9.5]
    SHM = interpolate.interp1d(Ms, Mh, fill_value = "extrapolate")
    mstarBOUND = Moster(11.5, 0)
    MLBOUND = (10**(mstarBOUND - 9))/Lum(galaxy_name)
    mstar = np.log10((ML)*Lum(galaxy_name)*(10**9))
    if ML > MLBOUND or model == 'NFW' or model == 'DC14':
        y = mstar
        return op.fsolve(Moster, 13, args=y) #13 is starting estimate
    elif ML < MLBOUND:
        return SHM(mstar)
    
def Mdm_NFW_calc(r_input,Mhalo_log,c):# Calculation for dark matter halo mass under the NFW model
    r = r_input
    Rvir = Rvir_calc(Mhalo_log)
    Rs = Rvir / c
    x = r / Rs # Line 488 From ForAnaCamila. Check whether r is under logarithm

    gx = np.log(1+x) -  (x / (1 + x))
    gc = np.log(1+c) - (c / (1 + c))
    Mdm_result = Mhalo_log + np.log10(gx) - np.log10(gc) # Line 490 from ForAnaCamila
    return Mdm_result

def Mdm_DC14_calc(r_input,Mhalo_log,c,ML): # Calculation for dark matter halo mass under the DC14 model
    Mdm_result = np.array([])
    Rvir= Rvir_calc(Mhalo_log)
    # DC14 calculations from Di Cinto et al 2014
    X = np.log10(ML*Lum(galaxy_name)*(10**9)) - Mhalo_log # Subtraction as the masses are under logarithm.
    if X<-4.1:
        X = -4.1    #extrapolation beyond accurate DC14 range
    if X>-1.3:
        X = -1.3
    alpha = 2.94 - np.log10( (10**((X+2.33)*-1.08)) + ((10**((X+2.33))*2.29)) )
    # alpha is the transition parameter between the inner slope and outer slope
    beta = 4.23 + (1.34* X) + (0.26 * (X**2)) # beta is the outer slope
    gamma = -0.06 - np.log10( (10**((X+2.56)*-0.68)) + (10**(X+2.56)) ) # gamma is the inner slope
        
    # alpha, beta and gamma are constrained as shown in Di Cinto et al. 2014
    
    c_sph = c * (1 + (0.00001 * math.exp(3.4 * (X + 4.5)))) # Equation 6, Di Cintio et al 2014 ///0.00003->0.00001?
    rm_2 = Rvir / c_sph
    r_s = rm_2 / (( (2 - gamma) / (beta - 2) )**(1 / alpha))
    temp1 = quad(lambda x: ((x**2) / (((x / r_s)**gamma) * (1+(x / r_s)**alpha)**((beta - gamma) / alpha) )), 0, Rvir)
    Ps = (10 ** Mhalo_log) / (4 * math.pi * temp1[0]) # intergration provides a tuple, thus we read the first result from the tuple as the desired result
    # Calculating Ps (scale density) as the value where M(Rvir) = Mhalo. From Di Cintio et al 2014.
    # using 'temp1' as a temporary variable to avoid syntax errors
    for i in range(len(r_input)): # The calculation must be iterated as the quad() function does not like
                                  # the use of np.array. This is inefficient and could be improved if quad()
                                  # is not used.
        r = r_input[i]
        
        temp1 = quad(lambda x: ((x**2) / (((x / r_s)**gamma) * (1+(x / r_s)**alpha)**((beta - gamma) / alpha) )), 0, r)
        a = np.append(Mdm_result, [np.log10(4 * math.pi * Ps * temp1[0])]) # a has to be used as np.append creates a new appended array
        Mdm_result = a
        """if np.any(np.isnan(Mdm_result))==True:
        print(ML, Mhalo_log, X)"""
    
    return Mdm_result

def opM(r_input,Mhalo_log,c,ML):
    global wdmh
    Mhalo_log = halo_calc(ML)
    popt, pcov = op.curve_fit(Vtot_calc, r_list(galaxy_name), Vtot_data(galaxy_name), sigma = Verr(galaxy_name), bounds = ([Mhalo_log-0.2,c,ML],[Mhalo_log + 0.2,c + 0.0000001,ML + 0.00000001]), max_nfev=10000, method = 'trf')
    wdmh = popt[0]
    return Vtot_calc(r_input, popt[0], popt[1], popt[2])

"Selects the function to use based on the value of 'model' and returns a value for Mdm."
"Then calculates the total velocity curve, Vc, using the two main equations in the introduction"

def Vdm_calc(r_input,Mhalo_log,c, ML):
    global wdmc
    
    if model == 'WDM1':
        c = c_calc(c_data1, Mhalo_log)
    elif model == 'WDM2':
        c = c_calc(c_data2, Mhalo_log)
    elif model == 'WDM3':
        c = c_calc(c_data3, Mhalo_log)
    '''else: # include this else function to restrict the mhalo-c relation to a gaussian distribution for NFW and DC14
        alpha = 10.84
        gamma = 0.085
        M_param = 5.5e17
        theory_mhalo_units = 10**Mhalo_log*1e-12*h
        c = 10**(np.log10((alpha *(1/theory_mhalo_units)**gamma)*(1+(theory_mhalo_units/(M_param*1e-12))**0.4))+wdm_dist)'''
    if model in ("NFW","WDM0","WDM1","WDM2","WDM3"):
        Mdm = Mdm_NFW_calc(r_input,Mhalo_log,c)
    elif model == 'DC14':
        Mdm = Mdm_DC14_calc(r_input,Mhalo_log,c,ML)
    else:
        Mdm = 0
        print('Error: model not known. Please select one specified in the list')
    Vdm = np.sqrt((G * 10**Mdm ) / r_input) # Equation 1, H. Katz et al. 2017
    wdmc = c
    return Vdm

dist = False # True to include the Mhalo-Mstar gaussian dist, false to leave it unrestricted

def Vtot_calc(r_input,Mhalo_log,c,ML):
    global g_Mhalo
    global g_ML
    if dist == True:
        theory_mhalo = np.linspace(Mhalo_log_lim[0], Mhalo_log_lim[1], 100) # make up a load of halos between limits
        theory_mstar = Moster(theory_mhalo, 0) # use mosters relationship to get the galax
        mstarBOUND = Moster(11.5, 0)
        MLBOUND = (10**(mstarBOUND - 9))/Lum(galaxy_name)
        if ML > MLBOUND or model == 'NFW' or model == 'DC14':
            grad = np.gradient(theory_mstar, 0.06)
            m = interpolate.interp1d(theory_mstar, grad, fill_value = "extrapolate")
        elif ML < MLBOUND:
            Ms = [7.5,7.8,8.1,8.5,9.1,9.5]
            grad = np.gradient(Ms, 0.25)
            m = interpolate.interp1d(Ms, grad, fill_value = "extrapolate")
        mstar = np.log10((ML)*Lum(galaxy_name)*(10**9))
        horiz = m(mstar) / np.sqrt(m(mstar)**2 + 1)
        vert = 1 / np.sqrt(m(mstar)**2 + 1)
        Mhalo_log = halo_calc(ML) - halo_dist*horiz
        g_Mhalo = Mhalo_log
        ML = (10**((mstar + halo_dist*vert) -9)) / Lum(galaxy_name)
        g_ML = ML

    Vtot = np.sqrt((Vdm_calc(r_input,Mhalo_log,c, ML)**2)+np.sign(Vgas(galaxy_name))*(Vgas(galaxy_name)**2)
    +(ML*(Vstar(galaxy_name)**2))+(ML*1.4*(Vbulge(galaxy_name)**2)))# Equation 9, H. Katz et al. 2017
    return Vtot

"Finds optimum values of Mhalo, concentration and ML for a curve with minimal squared residuals"
"""def opC(r_input,Mhalo_log,c,ML):
    global wdmc
    if model == 'WDM1':
        c = c_calc(c_data1, Mhalo_log)
    if model == 'WDM2':
        c = c_calc(c_data2, Mhalo_log)
    if model == 'WDM3':
        c = c_calc(c_data3, Mhalo_log)
    popt, pcov = op.curve_fit(Vtot_calc, r_list(galaxy_name), Vtot_data(galaxy_name), sigma = Verr(galaxy_name), bounds = ([Mhalo_log,10**(np.log10(c) - 0.2),ML],[Mhalo_log + 0.001,10**(np.log10(c) + 0.2),ML + 0.00000001]), max_nfev=10000, method = 'trf')
    wdmc = popt[1]
    return Vtot_calc(r_input, popt[0], popt[1], popt[2])"""

def rotCurve(galaxy, Mhalo_log, c, ML):
    """
    Plots the rotation curve for a specified set of data rather than using limits
    
    Inputs:
    
    galaxy_name - Name of the galaxy as in the data set
    Mhalo_log - base 10 log of the DM halo mass in solar masses
    c - concentration parameter (virial radius divided by scale radius)
    ML - Mass to light ratio
    
    No returned value
    
    """
    plt.plot(r_list(galaxy),Vtot_calc(r_list(galaxy), Mhalo_log, c, ML),label = "Total velocity fit")
    plt.plot(r_list(galaxy),Vdm_calc(r_list(galaxy), Mhalo_log, c, ML),label = "Modelled DM velocity")
    plt.plot(r_list(galaxy),Vtot_data(galaxy),'bo', label = "Total velocity (data)")
    plt.plot(r_list(galaxy),Vstar(galaxy),'ro',label = "Disk velocity")
    plt.plot(r_list(galaxy),Vbulge(galaxy),'mo',label = "Bulge velocity")
    plt.plot(r_list(galaxy),Vgas(galaxy),'go',label = "Gas velocity")
    plt.errorbar(r_list(galaxy), Vtot_data(galaxy), yerr=Verr, fmt=".k")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel("Radius - kpc")
    plt.ylabel("Velocity - km/s")
    plt.title(model)
    plt.show()
#==============================================================================
# Optimising rotation curve
#==============================================================================
def opline(galaxy, plot = True, save = True,show=False):
    global wdm_dist
    wdm_dist = np.random.normal(0,0.16)
    #Error Checking########################
    """
    print('Vgas:',len(Vgas(galaxy)),'Vstar:',len(Vstar(galaxy)),'Vtot_data:',
        len(Vtot_data(galaxy)),'Verr:',len(Verr(galaxy)))
    #from this ^^^ line of code comes the error of not being able to broadcast 1d matrixes of differing lengths
    #print(Vtot_data(galaxy))
    print(Vgas(galaxy))
    error_count = 0 # comment out if not testing
    for i in Vgas(galaxy):
        error_count +=1

    print('The Length of Vgas is:',len(Vgas(galaxy)),'The length of Vgas is:',error_count)
    """
    #######################################
    ML_lim[1] = min(((Vtot_data(galaxy)+Verr(galaxy))**2 - np.sign(Vgas(galaxy))*(Vgas(galaxy)**2))/(Vstar(galaxy))**2)
    global halo_dist
    halo_dist = np.random.normal(0,0.16)
    #if model == 'NFW' or model == 'DC14':
    popt, pcov = op.curve_fit(Vtot_calc, r_list(galaxy), Vtot_data(galaxy), sigma = Verr(galaxy), p0 = (11, 10, ML_lim[1]), bounds = ([Mhalo_log_lim[0],c_lim[0],ML_lim[0]],[Mhalo_log_lim[1],c_lim[1],ML_lim[1]]), max_nfev=20000, method = 'trf')
    #elif model in ('WDM1', 'WDM2', 'WDM3'):
        #popt, pcov = op.curve_fit(opM, r_list(galaxy), Vtot_data(galaxy), sigma = Verr(galaxy), bounds = ([wdmh,c_lim[0],ML_lim[0]],[wdmh+0.00001,c_lim[1],ML_lim[1]]), max_nfev=10000, method = 'trf')
    #if model in ('WDM1', 'WDM2', 'WDM3'): # change this if statement to all models to restrict the mhalo-c relation to a gaussian distribution for NFW and DC14
    popt[1] = wdmc
    """
    popt[0] = g_Mhalo # leave these 2 lines in to include the Mhalo-Mstar gaussian dist
    popt[2] = g_ML    # comment out to leave it unrestricted
    """
    # popt is an array of the optimal values for logMhalo, c and ML
    #pcov is the covarience matrix of Mhalo_log,c and ML
    # The argument in the op.curve_fit "bounds = ([],[])" are the bounds for the optimised values
    # The first array is the lower bounds, the second array is the upper bounds which I have left as +infinity

    print(popt)
    if plot == True:
        if save == False:

            plt.figure(dpi=120,figsize=(12,12))
            plt.plot(r_list(galaxy),Vtot_calc(r_list(galaxy), *popt),label = "Total velocity fit")
            plt.plot(r_list(galaxy),Vdm_calc(r_list(galaxy), *popt),label = "Modelled DM velocity")
            #plots the calculated total velocities as a curve
            print("Halo Mass:", popt[0])
            print("C:", popt[1])
            print("Mass-to-light ratio:",popt[2])
            print(pcov)
            print(np.sqrt(np.diag(pcov)))
            plt.plot(r_list(galaxy),Vtot_data(galaxy),'bo', label = "Total velocity (data)")
            #plots the total velocities from the data
            plt.plot(r_list(galaxy),np.sqrt(popt[2])*Vstar(galaxy),'ro',label = "Disk velocity")
            plt.plot(r_list(galaxy),np.sqrt(popt[2]*1.4)*Vbulge(galaxy),'mo',label = "Bulge velocity")
            plt.plot(r_list(galaxy),Vgas(galaxy),'go',label = "Gas velocity")
            plt.errorbar(r_list(galaxy), Vtot_data(galaxy), yerr=Verr(galaxy), fmt=".k")
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.xlabel("Radius - kpc")
            plt.ylabel("Velocity - km/s")
            plt.title(model+' '+galaxy)
            if show == True:

                plt.show()

        if save == True:

            [plt.figure(dpi=120,figsize=(12,12))]
            plt.plot(r_list(galaxy),Vtot_calc(r_list(galaxy), *popt),label = "Total velocity fit")
            plt.plot(r_list(galaxy),Vdm_calc(r_list(galaxy), *popt),label = "Modelled DM velocity")
            #plots the calculated total velocities as a curve
            print("Halo Mass:", popt[0])
            print("C:", popt[1])
            print("Mass-to-light ratio:",popt[2])
            print(pcov)
            print(np.sqrt(np.diag(pcov)))
            plt.plot(r_list(galaxy),Vtot_data(galaxy),'bo', label = "Total velocity (data)")
            #plots the total velocities from the data
            plt.plot(r_list(galaxy),np.sqrt(popt[2])*Vstar(galaxy),'ro',label = "Disk velocity")
            plt.plot(r_list(galaxy),np.sqrt(popt[2]*1.4)*Vbulge(galaxy),'mo',label = "Bulge velocity")
            plt.plot(r_list(galaxy),Vgas(galaxy),'go',label = "Gas velocity")
            plt.errorbar(r_list(galaxy), Vtot_data(galaxy), yerr=Verr(galaxy), fmt=".k")
            #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.legend(loc='upper right')
            plt.xlabel("Radius - kpc")
            plt.ylabel("Velocity - km/s")
            plt.title(model+' '+galaxy)
            plt.savefig(model+galaxy)
            if show == True:
                plt.show()

    return popt, max(Vtot_calc(r_list(galaxy), *popt)), pcov

def obtainMhaloMgal(galnames):
    return Mhalo,Mgal
#==============================================================================
# Relation plots
#==============================================================================
def acc(r_input,Mhalo_log,c,ML):
    if model in ("NFW","WDM1","WDM2","WDM3"):
        Mdm = Mdm_NFW_calc(r_input,Mhalo_log,c)
    elif model == 'DC14':
        Mdm = Mdm_DC14_calc(r_input,Mhalo_log,c,ML)
    V_bar = np.sqrt(np.sign(Vgas(galaxy_name))*(Vgas(galaxy_name)**2)+(ML*(Vstar(galaxy_name)**2))+(ML*1.4*(Vbulge(galaxy_name)**2)))
    g = ((G * (10**Mdm) / r_input**2) + ((V_bar**2) / r_input)) / 3.086e13 #unit conv
    return g

def acc_bar(r_input,ML):
    V_bar = np.sqrt(np.sign(Vgas(galaxy_name))*(Vgas(galaxy_name)**2)+(ML*(Vstar(galaxy_name)**2))+(ML*1.4*(Vbulge(galaxy_name)**2)))
    g = ((V_bar**2) / r_input) / 3.086e13 
    return g

def acc_obs(r_input):
    V_obs = Vtot_data(galaxy_name)
    g = ((V_obs**2) / r_input) / 3.086e13
    return g

def plotg(galaxy):  #Plots the radial acceleration profile for a single galaxy
    galaxy_name = galaxy
    varis = opline(galaxy_name,False)[0]
    Mhalo_log = varis[0]
    c = varis[1]
    ML = varis[2]
    plt.plot(r_list(galaxy),acc(r_list(galaxy), Mhalo_log, c, ML))
    plt.xlabel("Radius - kpc")
    plt.ylabel("Acceleration - m/s^2")
    plt.xscale('log')
    plt.yscale('log')
    plt.title(model + galaxy_name)
    plt.show()
    print((acc(r_list(galaxy), Mhalo_log, c, ML)))
    
def theory_g(x, g_scale):    #function fitting char(), McGaugh 2016
    return x / (1 - np.exp(- np.sqrt(x/g_scale))) 

lowmass = ['CamB', 'D512-2', 'D564-8', 'D631-7', 'DDO064', 'DDO154', 'F568-V1',  'F574-2', 'IC2574', 'KK98-251', 'NGC0247', 'NGC4068', 'PGC51017', 'UGC04483', 'UGC06628', 'UGC06667', 'UGC08837', 'UGC10310', 'UGCA442', 'UGCA444']

def theory_tf(x, m, b):
    return ((x - b + m) / m)

def Mhalo_c_Mstar_scatters(galaxy_names):
    "galaxy_names is an array of the names of the galaxies to be scatter plotted"
    global galaxy_name#global
    global wdm_dist
    global halo_dist
    g = np.array([])
    g_bar = np.array([])
    g_obs = np.array([])
    V = np.array([])
    M = np.array([])
    obs_V = np.array([])
    obs_M = np.array([])
    j = 0
    masses = np.array([])
    cVals = np.array([])
    Herr = np.array([])
    Cerr = np.array([])
    MLerr = np.array([])
    starmasses = np.array([])
    starluminosities = np.array([])
    centralLums = np.array([])
    chi_array = np.array([])
    theory_mhalo = np.linspace(Mhalo_log_lim[0], Mhalo_log_lim[1], 100) #create theoretical halos between the limits
    theory_cw = np.zeros_like(theory_mhalo)
    theory_cn = np.zeros_like(theory_mhalo)
    for i in range(len(galaxy_names)):
        galaxy_name = galaxy_names[i]
        varis = opline(galaxy_name, False) # uncheck to true to plot galaxy rotation curves
        popt, pcov = varis[0], varis[2] #coefficients and covariances from curve fit
        Mbar_calc = np.log10((10**9)*(popt[2]*Lum(galaxy_name)) + 1.33*H1Mass(galaxy_name))
        V = np.append(V, np.log10(varis[1]))
        M = np.append(M, Mbar_calc)
        if galaxy_name not in ['CamB', 'D512-2', 'D564-8', 'DDO064' , 'F574-2', 'NGC4068', 'PGC51017', 'UGC04483', 'UGC06628', 'UGC08837', 'UGCA281']:   #stops NAN error
                obs_V = np.append(obs_V, TF_V(galaxy_name))
                obs_M = np.append(obs_M, TF_M(galaxy_name))
        j = j+ 1
        maxgbar = np.nanmax(acc_bar(r_list(galaxy_name), popt[2]))
        i = np.where(acc_bar(r_list(galaxy_name), popt[2]) == maxgbar)    #takes values from same index
#        print(galaxy_name, i[0], acc_obs(r_list(galaxy)))
        maxg = acc(r_list(galaxy_name), popt[0], popt[1], popt[2])[i[0]]
        maxgobs = acc_obs(r_list(galaxy_name))[i[0]]
        if np.isnan(maxgbar) == True:
            maxgobs = np.nan
        #print(galaxy_name, i[0], maxgobs)
        g = np.append(g, maxg)
        g_bar = np.append(g_bar, maxgbar)
        g_obs = np.append(g_obs, maxgobs)
        masses = np.append(masses, popt[0])
        """opline(galaxy_name, True)
        print(opline(galaxy_name, False))
        print(popt)"""
        cVals = np.append(cVals, np.log10(popt[1]))
        Herr = np.append(Herr, np.sqrt(np.diag(pcov))[0])
        Cerr = np.append(Cerr,  np.sqrt(np.diag(pcov))[1])
        MLerr = np.append(MLerr, np.sqrt(np.diag(pcov))[2] )
        MLerr = np.log10((popt[2]+MLerr)*Lum(galaxy_name)*(10**9))- np.log10(popt[2]*Lum(galaxy_name)*(10**9))
        Cerr = np.log10(popt[1]+Cerr) - np.log10(popt[1])
        #print(Herr)
        chi_squared = np.sum(((Vtot_calc(r_list(galaxy_name), *popt)-Vtot_data(galaxy_name))/Verr(galaxy_name))**2)
        wdm_dist = 0
        if model == 'WDM1':
            theory_cw = np.log10(c_calc(c_data1, theory_mhalo))
        if model == 'WDM2':                                 #simplify/combine
            theory_cw = np.log10(c_calc(c_data2, theory_mhalo))
        if model == 'WDM3':
            theory_cw = np.log10(c_calc(c_data3, theory_mhalo))
        starmasses = np.append(starmasses, np.log10(abs(popt[2])*Lum(galaxy_name)*(10**9)))
        starluminosities = np.append(starluminosities, Lum(galaxy_name)/(10.**9.))
        #print(starmasses[-1])
        centralLums = np.append(centralLums, Bright(galaxy_name))
        if model == 'NFW' or model == 'DC14':
            reduced_chi_squared = (chi_squared)/(len(r_list(galaxy_name))-2)
        if model == 'WDM1' or model == 'WDM2' or model == 'WDM3':
            reduced_chi_squared = (chi_squared)/(len(r_list(galaxy_name))-2)
        chi_array = np.append(chi_array, reduced_chi_squared)
        #print (model, galaxy_name, reduced_chi_squared)
    plt.figure(1)
    plt.scatter(masses, cVals, c=centralLums, cmap='jet', vmax = 4.4)
    d = plt.errorbar(masses, cVals, xerr = Herr, yerr = Cerr, fmt='none', ecolor='0.7', elinewidth='0.5', marker=None, mew=0, linestyle = ':')
    #d[-1][0].set_linestyle(':') # if everything breaks uncomment this, guesswork at best oh god help me
    #d[-1][1].set_linestyle(':') # LOOK HERE ^&^^^^^^^^^^^^^
    clb = plt.colorbar()
    clb.set_label("$log_{10}(Central\ surface\ brightness)\ (L_{\odot}\ pc^{-2})$")
    theory_mhalo_units = 10**theory_mhalo*1e-12*h
    if redshift == 0.0:
        alpha = 10.84
        gamma = 0.085
        M_param = 5.5e17#schneider 2012
    elif redshift == 1.1:
        alpha = 5.9
        gamma = 0.1
        M_param = 4.4e16
    else:
        print('redshift error!')
    theory_cn = np.log10((alpha *(1/theory_mhalo_units)**gamma)*(1+(theory_mhalo_units/(M_param*1e-12))**0.4))
    if model == 'NFW' or model == 'DC14':
        plt.plot(theory_mhalo, theory_cn, color = "black")
    if model =='WDM1'or model =='WDM2'or model =='WDM3':
        plt.plot(theory_mhalo, theory_cn, color = "gray", linestyle = "--")
        plt.plot(theory_mhalo, theory_cw, color = "black")
    if model == 'DC14':
        plt.axvline(x = 12, color = 'gray', linestyle = '--')
    plt.xlim(8, 14)
    plt.ylim(0, 2)
    plt.xlabel("$log(M_{halo})$")
    plt.ylabel("$log(c)$")
    plt.title(model + " $M_{halo} - c$ relation")
    #plt.savefig(model + " $M_{halo} - c$ relation") # THIS DOESNT WORK FOR SOME ARBITRATY STUPID REASON
    "Mhalo-Mstar relation"
    plt.figure(2)
    plt.title("$M_{star} M_{halo} relationship$")
    plt.scatter(masses, starmasses, c=centralLums, cmap='jet', vmax = 4.4) #LOOK HERE FOR STARTING POINT DATA MASSES HALO/GALAXY
    d = plt.errorbar(masses, starmasses, xerr = Herr, yerr = MLerr, fmt='none', ecolor='0.7', elinewidth='0.5', marker=None, mew=0, linestyle = ':')
    ####################################
    #Writing Data to file 

    file = open(str(model)+'_MstarMhalo.dat','w')
    file.write(str(starmasses)+'\n'+str(masses))
    file.close()

    file = open(str(model)+'_concentration.dat','a')
    for i in range(len(masses)):
        file.write(str(masses[i])+' '+str(cVals[i])+' '+str(centralLums[i])+'\n')
    file.close()

    file = open(str(model)+'_MLratio.dat','a')
    for i in range(len(starmasses)):
        file.write(str(starmasses[i])+' '+str(starluminosities[i])+'\n')
    file.close()

    ####################################
    #d[-1][0].set_linestyle(':')
    #d[-1][1].set_linestyle(':')
    clb = plt.colorbar()
    clb.set_label("$log_{10}(Central\ surface\ brightness)\ (L_{\odot}\ pc^{-2})$")
    plt.xlim(8, 14)
    plt.ylim(6,12)
    plt.xlabel("$log(M_{halo})$")
    plt.ylabel("$log(M_{star})$")
    plt.title(model + " $M_{halo} - M_{star}$ relation")
    theory_mstar = Moster(theory_mhalo, 0)
    if model == 'DC14':
        bound1 = -4.1 + theory_mhalo
        bound2 = -1.3 + theory_mhalo
        plt.plot(theory_mhalo, bound1, color = "gray", linestyle = '--')
        plt.plot(theory_mhalo, bound2, color = "gray", linestyle = '--')
    if model =='WDM1'or model =='WDM2'or model =='WDM3':
        Mh = [10,10.25,10.5,10.75,11,11.25]
        Ms = [7.5,7.8,8.1,8.5,9.1,9.5]
        SHM = interpolate.interp1d(Mh, Ms, fill_value = "extrapolate")
        theory_mhalo1 = theory_mhalo[theory_mhalo < 11.5]
        plt.plot(theory_mhalo1, SHM(theory_mhalo1), color = 'red', linestyle = '--')
    plt.plot(theory_mhalo, theory_mstar, color = "black")
    plt.figure(3)
    alpha = 3.75
    alpha_error = 0.11
    beta = 9.5
    beta_error = 0.013
    pivot = 1.915
    theory_m = np.linspace(min(M), max(M), j)
    theory_v = ((theory_m - beta + (alpha * pivot)) / alpha)
    lower_v = ((theory_m-(beta + beta_error) + ((alpha - alpha_error) * pivot)) / (alpha + alpha_error))
    upper_v = ((theory_m-(beta - beta_error) + ((alpha + alpha_error) * pivot)) / (alpha - alpha_error))
    popt1, pcov1 = op.curve_fit(theory_tf, M, V)      #best line fit for data
    popt2, pcov2 = op.curve_fit(theory_tf, obs_M, obs_V)    #SPARC fit
    #chi_squared = np.sum(((theory_tf(M, *popt1)-theory_v)/(upper_v-lower_v))**2)
    #reduced_chi_squared = chi_squared/(len(galaxy_names)-len(popt1))
    plt.plot(theory_m, theory_v, color = "black", linestyle='--')
    plt.plot(M, theory_tf(M, *popt1), color = "blue")
    plt.plot(M, theory_tf(M, *popt2), color = "red")
    plt.fill_between(theory_m, lower_v, upper_v, facecolor = 'yellow')
    #plt.scatter(obs_M, obs_V)
    plt.scatter(M, V, c=centralLums, cmap='jet', vmax = 4.4)
    #eb1 = plt.errorbar(M, V, fmt='none', ecolor='black', elinewidth='0.5', marker=None, mew=0)
    #eb1[-1][0].set_linestyle(':')
    clb = plt.colorbar()
    clb.set_label("$log_{10}(Central\ surface\ brightness)\ (L_{\odot}\ pc^{-2})$")
    plt.xlim(min(M), max(M))
    plt.ylim(min(V), max(V))
    plt.xlabel('$log_{10}(M_{bar})\ (M_{\odot})$')
    plt.ylabel('$log_{10}(V)\ (kms^{-1})$')
    plt.title(model + " Tully-Fisher relation")
    plt.figure(4)
    theory_1to1 = np.linspace(10**-12.5, 10**-9, len(galaxy_names))
    popt, pcov = op.curve_fit(theory_g, g_bar, g, p0 = 1.2e-10) 
    plt.plot(theory_1to1, theory_g(theory_1to1, *popt))     #Our best fit
    plt.plot(theory_1to1, theory_1to1, color = "black", linestyle='--') #No DM, 1:1 line
    plt.plot(theory_1to1, theory_g(theory_1to1, 1.2e-10), color = "red")    #McGaugh fit 
    plt.scatter(g_bar, g, c=centralLums, cmap='jet', vmax = 4.4)
    clb = plt.colorbar()
    clb.set_label("$log_{10}(Central\ surface\ brightness)\ (L_{\odot}\ pc^{-2})$")
    plt.xlabel("$g_{bar} [m s^{-2}]$")
    plt.ylabel("$g [m s^{-2}]$")
    plt.xlim(10**-12.5, 10**-9)
    plt.ylim(10**-12, 10**-9)
    plt.xscale('log')
    plt.yscale('log')
    plt.title(model + " Characteristic Acceleration")
    plt.show()
    print(popt)
    plt.show()

    return chi_array

Do_stats = False

if Do_stats == True:

    model = "NFW"
    NFW_chi = Mhalo_c_Mstar_scatters(galaxy_names)
    NFW_chi = np.sort(NFW_chi)

    model = "DC14"
    DC14_chi = Mhalo_c_Mstar_scatters(galaxy_names)
    DC14_chi = np.sort(DC14_chi)

    model = "WDM1"
    WDM1_chi = Mhalo_c_Mstar_scatters(galaxy_names)
    WDM1_chi = np.sort(WDM1_chi)

    model = "WDM2"
    WDM2_chi = Mhalo_c_Mstar_scatters(galaxy_names)
    WDM2_chi = np.sort(WDM2_chi)

    model = "WDM3"
    WDM3_chi = Mhalo_c_Mstar_scatters(galaxy_names)
    WDM3_chi = np.sort(WDM3_chi)

    cdf = np.linspace(0, 1, num=len(galaxy_names))

    plt.semilogx(NFW_chi, cdf, label = "NFW")

    plt.semilogx(DC14_chi, cdf, label = "DC14")

    plt.semilogx(WDM1_chi, cdf, label = "WDM1")

    plt.semilogx(WDM2_chi, cdf, label = "WDM2")

    plt.semilogx(WDM3_chi, cdf, label = "WDM3")

    plt.xlabel("reduced chi squared")
    plt.ylabel("CDF")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title("Chi squared fit")
    plt.xlim(0.1, 100)
    plt.ylim(0, 1)
    print ("Median Reduced Chi Squared Values:")
    print ("NFW:  ",  np.median(NFW_chi))
    print ("DC14: ",  np.median(DC14_chi))
    print ("WDM1: ",  np.median(WDM1_chi))
    print ("WDM2: ",  np.median(WDM2_chi))
    print ("WDM3: ",  np.median(WDM3_chi))
    plt.show()

"""f = open('tab_3kev.dat',"r") #File name of concentration data
c_data = f.readlines() # Inputs each line as a section of an array
c200c = []
M200c = []
for n in range(len(c_data)):
    temp1 = c_data[n].split()
#        if (temp1[0:5]) == " 0.00": # If the redshift is zero then c200c and M200c are added to their arrays
    c200c.append(float(temp1[3]))
    M200c.append(np.log10(float(temp1[0])*10e12))
c200c = np.log10(c200c)
plt.scatter(M200c, c200c)
theory_mhalo = np.linspace(9, 14, 100)
theory_c = np.zeros_like(theory_mhalo)
theory_mhalo_units = np.log10(10**theory_mhalo*h*10e-12) #conversion to units used in Dutton & Macciò 2014
theory_c += 0.905 - 0.101*(theory_mhalo_units) # Dutton & Macciò 2014
plt.plot(theory_mhalo, theory_c, color = "black")
plt.ylim(0.0,2.0)
plt.xlim(7.0,14.0)
plt.show()"""

#==============================================================================
# MCMC
#==============================================================================
"Setting up likelihood function and priors for MCMC"
def lnlike(theta, r_input, Vtot_data, Verr):
    Mhalo_log, c, ML = theta
    model = Vtot_calc(r_list(galaxy_name),Mhalo_log,c,ML)
    inv_sigma2 = np.divide(1.0, np.power(Verr, 2))
    return -0.5*(   np.sum(    np.power(Vtot_data-model, 2) * inv_sigma2   )    )


def lnprior(theta):
    Mhalo_log, c, ML = theta
    if Mhalo_log_lim[0] < Mhalo_log < Mhalo_log_lim[1] and c_lim[0] < c < c_lim[1] and ML_lim[0] < ML < ML_lim[1]:
        return 0.0
    return -np.inf

def lnprob(theta, r_input, Vtot_input, Verr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, r_input, Vtot_input, Verr)

"The MCMC and corner plotting"

def MCMC_Fit():
    ndim, nwalkers = 3, 100
    popt = opline(galaxy_name,False)[0]
    pos = [popt + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = (r_list(galaxy_name), Vtot_data(galaxy_name), Verr(galaxy_name)))
    print("Running MCMC...")
    sampler.run_mcmc(pos, 1000, rstate0=np.random.get_state())
    print("Done.")
    plt.clf()
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 9))
    axes[0].plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
#    axes[0].yaxis.set_major_locator(MaxNLocator(5))
    axes[0].set_ylabel("$Mhalo_log$")
    axes[1].plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
#    axes[1].yaxis.set_major_locator(MaxNLocator(5))
    axes[1].set_ylabel("$c$")
    axes[2].plot(np.exp(sampler.chain[:, :, 2]).T, color="k", alpha=0.4)
#    axes[2].yaxis.set_major_locator(MaxNLocator(5))
    axes[2].set_ylabel("$ML$")
    axes[2].set_xlabel("step number")
    fig.tight_layout(h_pad=0.0)
    fig.savefig("line-time.png")
    sampler.run_mcmc(pos, 250)
    samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
    fig = corner.corner(samples, labels=["$Mhalo_log$", "$c$", "$ML$"],truths=[opline(galaxy_name,False)[0][0],opline(galaxy_name,False)[0][1],opline(galaxy_name,False)[0][2]])
    #fig.savefig("triangle.png")
    print(fig)
    print("Mean acceptance fraction: {0:.3f}"
                .format(np.mean(sampler.acceptance_fraction)))

def targeted_sampling(model,random=False,targeted=True,N=10,Model_DC14=True,Model_NFW=True):

    #selects galaxy names at random from listed galaxies then feeds its into plotting loop
    print(galaxy_names_plot)
    if random == True:

        for i in galaxy_names_plot:
            i = galaxy_name

            if model==DC14:
                
                
                #galaxy_name = input("Galaxy ID: ")

                opline(galaxy_name) # do not under any circumstances make this a list, array or 
                #anything plural, it will return a truth error. 
                #The truth value of an array with more than one element is ambiguous
                

            if Model==NFW:
                
                opline(galaxy_name)

    if targeted==True:
        if model == 'DC14':

            opline(galaxy_name) # do not under any circumstances make this a list, array or 
            #anything plural, it will return a truth error. 
            #The truth value of an array with more than one element is ambiguous


        if model == 'NFW':

            print('test')
            opline(galaxy_name)


    return

def wholesample(galdata,save=False):
    return [y[0] for y in [z.split() for z in [x for x in galdata]]]
    #prints every galaxy name in sample, first finds all lines in specified galdata
    #then splits the rows into collumns, and takes the 0th item in each row and 
    #assigns them to a list



#Model_NFW = True
#Model_DC14 = True
#Model_WDM1 = False
#Model_WDM2 = False
#Model_WDM3 = False

Model_MCMC = False # Set this to true to do MCMC

galaxy_names_plot = [galaxy_names[i] for i in [x for x in random.sample(range(0,len(galaxy_names)),10)]]



for i in wholesample(data_lines1):
    galaxy_name = i 
    opline(galaxy_name)


targeted_sampling(model)


if Model_MCMC == True:

    opline(galaxy_name, True)
    MCMC_Fit()
    model = 'WDM1'
    opline(galaxy_name, True)
    MCMC_Fit()
    galaxy_name = 'UGC04483'
    model = 'NFW'
    opline(galaxy_name, True)
    MCMC_Fit()
    model = 'DC14'
    opline(galaxy_name, True)
    MCMC_Fit()
    model = 'WDM1'
    opline(galaxy_name, True)
    MCMC_Fit()
    galaxy_name = 'UGC06628'
    model = 'NFW'
    opline(galaxy_name, True)
    MCMC_Fit()
    model = 'DC14'
    opline(galaxy_name, True)
    MCMC_Fit()
    model = 'WDM1'
    opline(galaxy_name, True)
    MCMC_Fit()
    print(opline(galaxy_name,False)[2])
    opline('D631-7')
    for i in range(len(under40)):
        galaxy_name=under40[i]
        opline(galaxy_name)
    MCMC_Fit()
    for i in range(len(under30)):
        galaxy_name = under30[i]
        print(galaxy_name)
        plt.figure(2*i+1)
        opline(galaxy_name, True)
        MCMC_Fit()
    model = 'DC14'#CamB returns NaN
    for i in range(len(under30)):
        galaxy_name = under30[i]
        print(galaxy_name)
        plt.figure(2*i+1)
        opline(galaxy_name, True)
        MCMC_Fit()
    model = 'WDM2'
    for i in range(len(under30)):
        galaxy_name = under30[i]
        print(galaxy_name)
        plt.figure(2*i+1)
        opline(galaxy_name, True)
        MCMC_Fit()
    model = 'WDM3'
    for i in range(len(under30)):
        galaxy_name = under30[i]
        print(galaxy_name)
        plt.figure(2*i+1)
        opline(galaxy_name, True)
        MCMC_Fit()
#==============================================================================
# Tully-Fisher
#==============================================================================
TF_names=['D631-7', 'DDO154', 'DDO161',
       'DDO168', 'DDO170', 'ESO079-G014', 'ESO116-G012', 'ESO563-G021',
       'F568-V1', 'F571-8', 'F574-1', 'F583-1', 'IC2574',
       'IC4202', 'KK98-251', 'NGC0024', 'NGC0055', 'NGC0247', 'NGC0289',
       'NGC0300', 'NGC0801', 'NGC0891', 'NGC1003', 'NGC1090', 'NGC2403',
        'NGC2841', 'NGC2903', 'NGC2915', 'NGC2976', 'NGC2998',
       'NGC3109', 'NGC3198', 'NGC3741', 'NGC3769',
       'NGC3877', 'NGC3893', 'NGC3917', 'NGC3949', 'NGC3953', 'NGC3972',
       'NGC3992', 'NGC4010', 'NGC4013', 'NGC4051', 'NGC4085',
       'NGC4088', 'NGC4100', 'NGC4138', 'NGC4157', 'NGC4183', 'NGC4217',
       'NGC5005',  'NGC5055', 'NGC5585', 'NGC5907',
       'NGC6015', 'NGC6195', 'NGC6503', 'NGC6674', 'NGC6946', 'NGC7331',
       'NGC7814', 'UGC00128', 'UGC00191', 'UGC00634',
       'UGC00731', 'UGC02259', 'UGC02487', 'UGC02885', 'UGC02916',
       'UGC02953', 'UGC03205', 'UGC03580', 'UGC04325',
       'UGC04499', 'UGC05253', 'UGC05716', 'UGC05721',
       'UGC05986', 'UGC06399', 'UGC06446', 'UGC06614',
       'UGC06667', 'UGC06786', 'UGC06787', 'UGC06818', 'UGC06917',
       'UGC06923', 'UGC06930', 'UGC06973', 'UGC06983', 'UGC07125',
       'UGC07151', 'UGC07399', 'UGC07524', 'UGC07603', 'UGC07690',
       'UGC08286', 'UGC08490', 'UGC08550',
        'UGC09133', 'UGC10310', 'UGC11455',
       'UGC12506', 'UGC12632', 'UGCA442', 'UGCA444']



def tf(galaxy_names):
    V = np.array([])
    M = np.array([])
    obs_V = np.array([])
    obs_M = np.array([])
    j = 0
    #Qerr = np.array([])
    centralLums = np.array([])
    for i in range(len(galaxy_names)):    
        global galaxy_name                #global!
        galaxy_name = galaxy_names[i]
        varis = opline(galaxy_name,False)
        ML = varis[0][2]
        Mbar_calc = np.log10((10**9)*(ML*Lum(galaxy_name)) + 1.33*H1Mass(galaxy_name))
        V = np.append(V, np.log10(varis[1]))
        M = np.append(M, Mbar_calc)
        centralLums = np.append(centralLums, Bright(galaxy_name))
        if galaxy_name not in ['CamB', 'D512-2', 'D564-8', 'DDO064' , 'F574-2', 'NGC4068', 'PGC51017', 'UGC04483', 'UGC06628', 'UGC08837', 'UGCA281']:   #stops NAN error
                obs_V = np.append(obs_V, TF_V(galaxy_name))
                obs_M = np.append(obs_M, TF_M(galaxy_name))
        j = j+ 1
        '''
        Mhalo_log = varis[0][0]
        c = varis[0][1]
        ML = varis[0][2]
        opV = varis[1]
        Rvir = Rvir_calc(Mhalo_log)
        Rs = 0.7 * Rvir / c
        if Rs > r_list(galaxy_name)[-1] or Rs < r_list(galaxy_name)[0]:
            print(galaxy_name, 'Out of Range')
            continue
        else:
            print ("yay")
            V_r = interpolate.interp1d(r_list(galaxy_name),opV)
            #print(galaxy_name, Rs)
            Mbar_calc = np.log10((10**9)*(ML*Lum(galaxy_name)) + 1.33*H1Mass(galaxy_name))
            V = np.append(V, np.log10(V_r(Rs)))
            M = np.append(M, Mbar_calc)
            if galaxy_name not in ['CamB', 'D512-2', 'D564-8', 'DDO064' , 'F574-2', 'NGC4068', 'PGC51017', 'UGC04483', 'UGC06628', 'UGC08837', 'UGCA281']:   #stops NAN error
                obs_V = np.append(obs_V, TF_V(galaxy_name))
                obs_M = np.append(obs_M, TF_M(galaxy_name))
            #Qerr = np.append(Qerr, 0.434* e_Lum(galaxy_name) / Lum(galaxy_name))
            centralLums = np.append(centralLums, Bright(galaxy_name))
            j=j+1'''
    alpha = 3.75
    alpha_error = 0.11
    beta = 9.5
    beta_error = 0.013
    pivot = 1.915
    theory_m = np.linspace(5.3, 11.5, j)
    theory_v = ((theory_m - beta + (alpha * pivot)) / alpha)
    lower_v = ((theory_m-(beta + beta_error) + ((alpha - alpha_error) * pivot)) / (alpha + alpha_error))
    upper_v = ((theory_m-(beta - beta_error) + ((alpha + alpha_error) * pivot)) / (alpha - alpha_error))
    popt1, pcov1 = op.curve_fit(theory_tf, M, V)      #best line fit for data
    popt2, pcov2 = op.curve_fit(theory_tf, obs_M, obs_V)    #SPARC fit
    #chi_squared = np.sum(((theory_tf(M, *popt1)-theory_v)/(upper_v-lower_v))**2)
    #reduced_chi_squared = chi_squared/(len(galaxy_names)-len(popt1))
    plt.plot(theory_m, theory_v, color = "black", linestyle='--')
    plt.plot(M, theory_tf(M, *popt1), color = "blue")
    plt.plot(M, theory_tf(M, *popt2), color = "red")
    plt.fill_between(theory_m, lower_v, upper_v, facecolor = 'yellow')
    #plt.scatter(obs_M, obs_V)
    plt.scatter(M, V, c=centralLums, cmap='jet', vmax = 4.4)
    #eb1 = plt.errorbar(M, V, fmt='none', ecolor='black', elinewidth='0.5', marker=None, mew=0)
    #eb1[-1][0].set_linestyle(':')
    clb = plt.colorbar()
    clb.set_label("$log_{10}(Central\ surface\ brightness)\ (L_{\odot}\ pc^{-2})$")
    plt.xlim(5.3, 11.5)
    plt.ylim(1.1, 2.6)
    plt.xlabel('$log_{10}(M_{bar})\ (M_{\odot})$')
    plt.ylabel('$log_{10}(V)\ (kms^{-1})$')
    plt.title(model + " Tully-Fisher relation")
    plt.show()
    #print(M, theory_tf(M, *popt1))
    #print(popt1, j, max(M), min(M))
#==============================================================================
# Acceleration
#==============================================================================

def char(galaxy_names):
    g = np.array([])
    g_bar = np.array([])
    g_obs = np.array([])
    centralLums = np.array([])
    for i in range(len(galaxy_names)):    
        galaxy = galaxy_names[i]
        global galaxy_name
        galaxy_name = galaxy
        #print(galaxy_name)
        varis = opline(galaxy_name,False)[0]
        Mhalo_log = varis[0]
        c = varis[1]
        ML = varis[2]
        maxgbar = np.nanmax(acc_bar(r_list(galaxy), ML))
        i = np.where(acc_bar(r_list(galaxy), ML) == maxgbar)    #takes values from same index
#        print(galaxy_name, i[0], acc_obs(r_list(galaxy)))
        maxg = acc(r_list(galaxy), Mhalo_log, c, ML)[i[0]]
        maxgobs = acc_obs(r_list(galaxy))[i[0]]
        if np.isnan(maxgbar) == True:
            maxgobs = np.nan
        #print(galaxy_name, i[0], maxgobs)
        g = np.append(g, maxg)
        g_bar = np.append(g_bar, maxgbar)
        g_obs = np.append(g_obs, maxgobs)
        centralLums = np.append(centralLums, Bright(galaxy_name))
    #print(len(g),len(g_bar),len(g_obs))
    #print(g_obs)
    theory_1to1 = np.linspace(10**-12.5, 10**-9, len(galaxy_names))
    popt, pcov = op.curve_fit(theory_g, g_bar, g, p0 = 1.2e-10) 
    plt.plot(theory_1to1, theory_g(theory_1to1, *popt))     #Our best fit
    plt.plot(theory_1to1, theory_1to1, color = "black", linestyle='--') #No DM, 1:1 line
    plt.plot(theory_1to1, theory_g(theory_1to1, 1.2e-10), color = "red")    #McGaugh fit 
    plt.scatter(g_bar, g, c=centralLums, cmap='jet', vmax = 4.4)
    clb = plt.colorbar()
    clb.set_label("$log_{10}(Central\ surface\ brightness)\ (L_{\odot}\ pc^{-2})$")
    plt.xlabel("$g_{bar} [m s^{-2}]$")
    plt.ylabel("$g [m s^{-2}]$")
    plt.xlim(10**-12.5, 10**-9)
    plt.ylim(10**-12, 10**-9)
    plt.xscale('log')
    plt.yscale('log')
    plt.title(model + " Characteristic Acceleration")
    plt.show()
    print(popt)
#==============================================================================
# Choose output
#==============================================================================
#tf(TF_names)
test = ['D631-7', 'DDO154', 'DDO161','UGC02953']
#char(galaxy_names)
#opline(galaxy_name, True)
#MCMC_Fit()
#Mhalo_c_Mstar_scatters(galaxy_names)
#rotCurve(galaxy_name,11.907,16, 0.1)
'''char(galaxy_names)
model = 'DC14'
char(galaxy_names)
model = 'WDM1'
char(galaxy_names)
model = 'WDM2'
char(galaxy_names)
model = 'WDM3'
char(galaxy_names)'''
stop = timeit.default_timer()
print("Time=",(stop - start)/60)
