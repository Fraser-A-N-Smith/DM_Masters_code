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

#r_list = np.array([0.72,1.43,2.16,2.87,3.59,4.3,5.03,5.75,6.46,7.18,7.91,8.62,11.49,14.3,17.19,20.07,22.96,25.85,28.74,31.62,34.51,37.4,40.29,43.17,45.92,48.81,51.7,54.59]) # The radii data
#Vtot_data = np.array([125.0,162.0,178.0,193.0,192.0,190.0,195.0,201.0,204.0,204.0,206.0,206.0,206.0,206.0,203.0,200.0,194.0,188.0,184.0,182.0,180.0,181.0,180.0,179.0,179.0,179.0,174.0,172.0])# The total velocity curve data
#Verr = np.array([17.6,8.5,5.22,1.88,3.59,0.97,0.99,0.45,0.81,0.65,0.66,1.54,1.9,0.42,4.19,1.23,4.09,5.2,1.16,2.47,3.85,8.43,4.55,0.44,1.96,2.99,3.39,4.84])
#Vstar = np.array([201.48,240.13,263.27,280.74,273.32,272.18,273.73,276.2,279.72,275.62,271.43,267.09,245.58,225.03,208.52,190.41,175.86,164.23,154.38,146.2,139.3,133.34,128.12,123.52,119.59,115.83,112.43,109.32])
#Vgas = np.array([2.2,3.72,4.09,5.34,6.03,5.11,8.42,14.42,19.11,22.51,25.8,28.49,33.07,45.73,47.26,44.56,40.45,37.41,38.46,40.22,42.57,40.57,41.86,42.92,44.68,44.56,43.74,41.74])
"Select input variables"

# Testing using D564-8, file name "DDO64" in SPARC folder

"Choose model; NFW or DC14 or SIDM or WDM:"
# Write the name of the model here. Ensure that it is written exactly as in the list



model = 'NFW'

model = input("Choose model; NFW or DC14 or SIDM or WDM:")

"""

'NFW' - NFW
'DC14' - Di Cinto et al. 2014
'SIDM' - Self-interacting
'WDM0' - Warm Dark Matter (not sure if this is needed)
'WDM1' - Warm Dark Matter, thermal with m_X = 3keV
'WDM2' - Warm Dark Matter, m_nu = 7keV, sin^2(2theta)=210^-10
'WDM3' - Warm Dark Matter, m_nu = 7keV, sin^2(2theta)=510^-11
"""

"Limits for optimal values of Mhalo_log, concentration and Mass-to-Light ratio"
"Variable_lim = np.array([lower bound,upper bound])"

Mhalo_log_lim = np.array([9,math.inf])
c_lim= np.array([1, math.inf])
ML_lim = np.array([0.3,0.8])

"Are the stellar and gas velocities provided?"
Vprovided = True # If Vstar and Vgas are given from data then set as 'True'. If not, set as 'False'

Mstar_log = 4
Mgas_log = 4
Rd = 1
al_gas = 1
Ms = 5
# Ms is the input free-streaming  scale for the WDM model

h = 0.7

galaxy_name = "NGC5055"

galaxy_name = input("Galaxy ID, e.g. NGC5055:")

r_list = np.array([])
Verr = np.array([])
Vstar = np.array([])
Vgas = np.array([])
Vtot_data = np.array([])


 
"Reads data from the SPARC file"

f = open("rotationCurvesSPARC.txt", "r")
data_lines = f.readlines()
r_list = np.array([])
Verr = np.array([])
Vstar = np.array([])
Vgas = np.array([])
Vtot_data = np.array([])
for i in range(len(data_lines)):
    temp2 = data_lines[i]
    if str(temp2[0:7]) in (galaxy_name,galaxy_name+" ",galaxy_name+"  "):
        temp3 = np.append([r_list],float(temp2[19:25]))
        r_list = temp3
            
        temp3 = np.append([Vtot_data],float(temp2[26:31]))
        Vtot_data = temp3
            
        temp3 = np.append([Verr],float(temp2[ 32: 38]))
        Verr = temp3
            
        temp3 = np.append([Vgas],float(temp2[ 39:45 ]))
        Vgas = temp3
            
        temp3 = np.append([Vstar],float(temp2[ 46:52]))
        Vstar = temp3

"""
Hello! I am Nicholas and I have been working on this code for 4 weeks under supervision of Francesco.

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

CURRENT ISSUES WITH THE CODE:

from scipy import interpolate
def c_calc():
    Mhalo_sun=Mhalo_log + np.log10(h)
    f = open("concentration_francesco_median.txt","r") #File name of concentration data
    c_data = f.readlines() # Inputs each line as a section of an array
    c200c = []
    M200c = []
    for n in range(len(c_data)):
        temp1 = c_data[n]
        if (temp1[0:5]) == " 0.00": # If the redshift is zero then c200c and M200c are added to their arrays
            c200c.append(float(temp1[24:29]))
            M200c.append(np.log10(float(temp1[14:22])))
    MtoC = interpolate.interp1d(M200c,c200c) # Declaring the Mass-to-Concentration interpolation
    # Approximates the value of c at our galaxy mass via interpolation of known data
    #c=10.^(logc+0.16*randomn(seed,n_elements(Mhs)))
    interpolated_c = MtoC(Mhalo_sun)
    #interpolates c for our value of Mhalo
    c_log = np.log10(interpolated_c)
    c_result = 10 ** (np.random.lognormal(c_log,0.16,1)) #(mean, SD., dimensions)
    # c is taken randomly from a lognormal distribution with log(c) as the mean
    f.close()
    return c_result
    
    2. Currently, the DC14 and SIDM functions are not returning suitable results that plot an appropriate
graph. I suspect that this is due to incorrect calculation.

    3. You will need to install the relevant packages, which are all imported at the start

    4. I originally coded using bog standard lists to act as arrays. I ran into trouble with the optimize
functions as they use 'np.array's so I changed to these instead. Not really an issue, just a note to
ensure that you stick with the numpy arrays rather than lists.

    5. At time of writing I have not been able to add 3 additional models that I would like to add.
These are:
    
    1- a thermal WDM model with m_X=3 keV
    2 - two sterile neutrino models with m_nu=7 keV and sin^2(2theta)=210^-10 and sin^2(2theta)=510^-11. 
    These approximatively correspond to the errorbar of the tentative 3.5 keV line. 
    
If I have not yet added these then please contact Francesco or Nicola Menci (nicola.menci@oa-roma.inaf.it)
to obtain the halo mass function N(M) at z = 0 and the concentration-mass relation c(M).

These 3 models should be able to just be implemented into the NFW calculation and changing the concentration
using the concentration-mass relation, similar to the way that the current WDM model has been implemented.

    6. The code does not read data from files, but instead uses np.arrays as input above. I tried to add
code that does this, but the provided data comes in different sets of .txt files which have formatted
the data in different ways which is a bit frustrating.

    7. Probably not an issue, but the fitted velocity curve is not a curve but a series of straight lines
from one radius to the next. I think that it would be hard to smooth this out as the Vgas and Vstar arrays
are sometimes inputs.

If you have any questions about the code then you can contact me via email at: nmitchell.62791@farnborough.ac.uk
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

"Defining subfunctions"

def Vstar_calc(): # Velocity of the stellar disk if not observed
    Vstar = np.array([])
    for i in range(len(r_list)):
        r = r_list[i]
        a = r / Rd # Using a as a temporary variable
        a_s = a * 0.5 # a_s is half of a
        B = ( iv(a_s,0) * kv(a_s,0) ) - ( iv(a_s,1) * kv(a_s,1) ) # iv and kv are modified Bessel functions
        V = math.sqrt( (0.5 * G * a * a * B * (10 ** Mstar_log)) / Rd) # Unsure about the total function in line 455 of ForAnaCamila
        a = np.append(Vstar,[V])
        Vstar = a
    return Vstar
def Vgas_calc(): # Velocity of gas disk if not observed
    Vgas = np.array([])
    for i in range(len(r_list)):
        r = r_list[i]
        Rd_gas = al_gas * Rd # al_gas is a coefficient for the assumed relationship between the scale-lengths of the
                         # stellar and gas disks. It is generally between 1 and 3
        a = r / Rd_gas # Using a as a temporary variable
        a_s = a * 0.5 # a_s is half of a
        B = ( iv(a_s,0) * kv(a_s,0) ) - ( iv(a_s,1) * kv(a_s,1) ) # iv and kv are modified Bessel functions
        V = math.sqrt( (0.5 * G * a * a * B * (10 ** Mgas_log)) / Rd_gas) # Unsure about the total function in line 455 of ForAnaCamila
        a = np.append(Vgas,[V])
        Vgas = a
    return Vgas
if Vprovided != True:
    Vstar = Vstar_calc()
    Vgas = Vgas_calc()
def Rvir_calc(Mhalo_log):
    z = 0.0 # Line 241 from ForAnaCamila
    omega_m = 0.3 # Line 242 from ForAnaCamila
    HH = 0.1 * h * math.sqrt((omega_m * ((1+z)**3)) + (1-omega_m)) # Line 410,411 from ForAnaCamila
    rho_c = (3 * (HH**2)) / (8 * math.pi * G) # Line 413 from ForAnaCamila
    k = (4*math.pi) / 3 # Line 414 from ForAnaCamila
    Rvir_result = np.cbrt(((10 ** Mhalo_log) / rho_c / k / 200)) # Check order of operations
    return Rvir_result
def Mdm_SIDM_calc(r_input,Mhalo_log,c):
    # Taken from 'ForAnaCamila' code, starting line 545
   r = r_input
   rho_b=np.log10(0.029)-0.19*(Mhalo_log-10)+9
   Rvir=Rvir_calc(Mhalo_log)
   rs=Rvir/c
   r_b=np.log10(rs*0.71*((rs*0.1)**(-0.08)))
   s_1=10**np.log10(r)
   s_2=s_1/(10**r_b)
   Mh0=np.log10(1.6)+rho_b+(3*r_b)
   Mdm_result=np.log10(4)+Mh0+np.log10(np.log(1.+s_2)-(np.arctan(s_2)))+(0.5*np.log(1+s_2**2))
#   print(Mdm_result)
#   Mdm_result = 1
   return Mdm_result
def c_calc(c_file, Mhalo_log):
    Mhalo_sun=Mhalo_log + np.log10(h)
    f = open(c_file,"r") #File name of concentration data
    c_data = f.readlines() # Inputs each line as a section of an array
    c200c = []
    M200c = []
    for n in range(len(c_data)):
        temp1 = c_data[n]
#        if (temp1[0:5]) == " 0.00": # If the redshift is zero then c200c and M200c are added to their arrays
        c200c.append(float(temp1[39:47]))
        M200c.append(float(temp1[0:13])) #!!!not sure if log10 should be used here!!!
    MtoC = interpolate.interp1d(M200c,c200c) # Declaring the Mass-to-Concentration interpolation
    # Approximates the value of c at our galaxy mass via interpolation of known data
    #c=10.^(logc+0.16*randomn(seed,n_elements(Mhs)))
    interpolated_c = MtoC(Mhalo_sun)
    #interpolates c for our value of Mhalo
    c_log = np.log10(interpolated_c)
    c_result = 10 ** (np.random.lognormal(c_log,0.16,1)) #(mean, SD., dimensions)
    # c is taken randomly from a lognormal distribution with log(c) as the mean
    f.close()
    return c_result
def Mdm_NFW_calc(r_input,Mhalo_log,c):# Calculation for dark matter halo mass under the NFW model
    if model == 'WDM0': # Converts the concentration if WDM using the relation from  Schneider et al 2012
        temp1 = c
        Mhl = 10 + np.log10(3.4)
        c = temp1 * ((1 + (15*(10**(Mhl - Ms))))**-0.3) # Equation for m_X = 1keV
    if model == 'WDM1':
        c = c_calc('tab_3kev.dat', Mhalo_log)
    if model == 'WDM2':
        c = c_calc('tab_RP_7keV_2e-10.dat', Mhalo_log)
    if model == 'WDM3':
        c = c_calc('tab_RP_7keV_5e-11.dat', Mhalo_log)
    r = r_input
    Rvir = Rvir_calc(Mhalo_log)
    Rs = Rvir / c
    x = r / Rs # Line 488 From ForAnaCamila. Check whether r is under logarithm
    gx = np.log(1+x) -  (x / (1 + x))
    gc = np.log(1+c) - (c / (1 + c))
    Mdm_result = Mhalo_log + np.log10(gx) - np.log10(gc) # Line 490 from ForAnaCamila
    return Mdm_result

def Mdm_DC14_calc(r_input,Mhalo_log,c): # Calculation for dark matter halo mass under the DC14 model
    Mdm_result = np.array([])    
    for i in range(len(r_input)): # The calculation must be iterated as the quad() function does not like
                                  # the use of np.array. This is inefficient and could be improved if quad()
                                  # is not used.
        r = r_input[i]
        # DC14 calculations from Di Cinto et al 2014
        X = Mstar_log - Mhalo_log # Subtraction as the masses are under logarithm.
        alpha = 2.94 - np.log10( (10**((X+2.33)*-1.08)) + ((10**((X+2.33))*2.29)) )
        # alpha is the transition parameter between the inner slope and outer slope
        beta = 4.23 + (1.34* X) + (0.26 * (X**2)) # beta is the outer slope
        gamma = -0.06 + np.log10( (10**((X+2.56)*-0.68)) + (10**(X+2.56)) ) # gamma is the inner slope
        
        # alpha, beta and gamma are constrained as shown in Di Cinto et al. 2014
        
        Rvir= Rvir_calc(Mhalo_log)
        temp1 = quad(lambda x: ((x**2) / (((x / Rd)**gamma) * (1+(x / Rd)**alpha)**((beta - gamma) / alpha) )), 0, Rvir)
        Ps = Mhalo_log / (4 * math.pi * temp1[0]) # intergration provides a tuple, thus we read the first result from the tuple as the desired result
        # Calculating Ps (scale density) as the value where M(Rvir) = Mhalo. From Di Cintio et al 2014.
        "Ask Francesco using log of Mhalo here or not?"
        # using 'temp1' as a temporary variable to avoid syntax errors
        c_sph = c * (1 + (0.00001 * math.exp(3.4 * (X + 4.5)))) # Equation 6, Di Cintio et al 2014
        rm_2 = Rvir / c_sph
        r_s = rm_2 / (( (2 - gamma) / (beta - 2) )**(1 / alpha))
        temp1 = quad(lambda x: ((x**2) / (((x / r_s)**gamma) * (1+(x / r_s)**alpha)**((beta - gamma) / alpha) )), 0, r)
        a = np.append(Mdm_result, [4 * math.pi * Ps * temp1[0]]) # a has to be used as np.append creates a new appended array
        Mdm_result = a
    # M(r) of DC14 model. Equation 5 from Di Cintio et al 2014
    return Mdm_result

"Selects the function to use based on the value of 'model' and returns a value for Mdm."
"Then calculates the total velocity curve, Vc, using the two main equations in the introduction"

def Vdm_calc(r_input,Mhalo_log,c,ML):
    if model in ("NFW","WDM0","WDM1","WDM2","WDM3"):
        Mdm = Mdm_NFW_calc(r_input,Mhalo_log,c)
    elif model == 'DC14':
        Mdm = Mdm_DC14_calc(r_input,Mhalo_log,c)
    elif model == 'SIDM':
        Mdm = Mdm_SIDM_calc(r_input,Mhalo_log,c)
    else:
        Mdm = 0
        print('Error: model not known. Please select one specified in the list')
    Vdm = np.sqrt((G * 10**Mdm ) / r_input) # Equation 1, H. Katz et al. 2017
    return Vdm #Should this be 10**Mdm or just Mdm ?

def Vtot_calc(r_input,Mhalo_log,c,ML):
    Vtot = np.sqrt((Vdm_calc(r_input,Mhalo_log,c,ML)**2)+(Vgas**2)+(ML*(Vstar**2)))# Equation 9, H. Katz et al. 2017
    return Vtot

"Finds optimum values of Mhalo, concentration and ML for a curve with minimal squared residuals"

def opline():
    "op.curve_fit is an optimisation function, but we should check that it is suitable for this purpose"
    popt, pcov = op.curve_fit(Vtot_calc, r_list, Vtot_data, bounds = ([Mhalo_log_lim[0],c_lim[0],ML_lim[0]],[Mhalo_log_lim[1],c_lim[1],ML_lim[1]]),maxfev=10000)
    # popt is an array of the optimal values for logMhalo, c and ML
    #pcov is the covarience matrix of Mhalo_log,c and ML
    # The argument in the op.curve_fit "bounds = ([],[])" are the bounds for the optimised values
    # The first array is the lower bounds, the second array is the upper bounds which I have left as +infinity
    plt.plot(r_list,Vtot_calc(r_list, *popt),label = "Total velocity fit")
    plt.plot(r_list,Vdm_calc(r_list, *popt),label = "Modelled DM velocity")
    #plots the calculated total velocities as a curve
    print("logMhalo:",popt[0])
    print("Concentration:",popt[1])
    print("Mass-to-light ratio:",popt[2])
    plt.plot(r_list,Vtot_data,'bo', label = "Total velocity (data)")
    #plots the total velocities from the data
    plt.plot(r_list,popt[2]**0.5*Vstar,'ro',label = "Stellar velocity")
    plt.plot(r_list,Vgas,'go',label = "Gas velocity")
    plt.errorbar(r_list, Vtot_data, yerr=Verr, fmt=".k")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel("Radius - kpc")
    plt.ylabel("Velocity - km/s")
    plt.title(model)
    plt.show()
    return popt

"Setting up likelihood function and priors for MCMC"

def lnlike(theta, r_input, Vtot_data, Verr):
    Mhalo_log, c, ML = theta
    model = Vtot_calc(r_input,Mhalo_log,c,ML)
    inv_sigma2 = np.divide(1.0, np.power(Verr, 2)) #+ model**2*np.exp(2*lnf)) ###do this
    return -0.5*(   np.sum(    np.power(Vtot_data-model, 2) * inv_sigma2 - np.log(inv_sigma2)     )    )


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
    popt = opline()
    pos = [popt + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = (r_list, Vtot_data, Verr))
    print("Running MCMC...")
    sampler.run_mcmc(pos, 500, rstate0=np.random.get_state())
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
    sampler.run_mcmc(pos, 500)
    samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
    fig = corner.corner(samples, labels=["$Mhalo_log$", "$c$", "$ML$"])
    fig.savefig("triangle.png")
    
opline()