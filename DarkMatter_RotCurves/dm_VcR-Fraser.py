#Theoretical Vc vs R, for various dark matter models.

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

#Cosmo = cosmology.getCurrent()
#h = Cosmo.h
h=0.7

cosmo = cosmology.setCosmology("planck18")
font_names = [f.name for f in fm.fontManager.ttflist]
#print(font_names)

cosmo = cosmology.setCosmology("planck18")

# Edit the font, font size, and axes widthmpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 18 # changes font size
plt.rcParams['axes.linewidth'] = 2 # changes linewidth

f = open("rotationCurvesSPARC.txt", "r")
data_lines = f.readlines()
f.close()
f = open("galaxymassesSPARC.txt", "r")
data_lines1 = f.readlines()
f.close()
f = open("centralLumSPARC.txt", "r")
data_lines2 = f.readlines()
f.close()


#Import AM data from Hao Fu resembling Katz et.al. 2016 paper

AM_HaoFu2020 = pd.read_csv('baldry_Bernardi_SMHM.txt',header=None,delim_whitespace=True)
AM_HaoFu2020_Halos = [x for x in pd.Series(AM_HaoFu2020[0]).values][2:] # remove the top two rows
AM_HaoFu2020_Halos = [float(x) for x in AM_HaoFu2020_Halos] # convert the strings into floats
AM_HaoFu2020_galaxies = [x for x in pd.Series(AM_HaoFu2020[1]).values][2:] # remove the top two rows
AM_HaoFu2020_galaxies = [float(x) for x in AM_HaoFu2020_galaxies] # convert the strings into floats

ShankarHalo = pd.read_csv('shankarhalos2021.csv',header=None,delim_whitespace=True)
ShankarHalo_M = [float(x) for x in pd.Series(ShankarHalo[0]).values]
print(ShankarHalo_M)


f = open("rotationCurvesSPARC.txt", "r")
data_lines = f.readlines()
f.close()
f = open("galaxymassesSPARC.txt", "r")
data_lines1 = f.readlines()
f.close()
f = open("centralLumSPARC.txt", "r")
data_lines2 = f.readlines()
f.close()

#Import SMF provided by Hao:
SMF = pd.read_csv('baldry_bernardi_SMF.txt',header=None,delim_whitespace=True)
SMF_logMgalMstar = [x for x in pd.Series(SMF[0]).values] #log(Mstar/Msun) for stellar mass function
SMF_logphigal = [np.log10(x) for x in pd.Series(SMF[1]).values] #  log(phi(Mstar)) [dex^-1 Mpc^-3]


#Plotting SMF:
plt.figure(figsize=(12,12),dpi=60) # Adjust figure parameters
plt.plot(SMF_logMgalMstar,SMF_logphigal) # plot log(PhiGal) against log(Mgal/Mstar)
plt.scatter(SMF_logMgalMstar,SMF_logphigal) # plot scatter points over line to illustrate data
plt.title('SMF from Baldry and Bernardi',fontname='DejaVu Sans')
plt.xlabel('log(M$_{*}$/M$_\odot$',fontname='DejaVu Sans')
plt.ylabel('log($\Phi_{gal}$)',fontname='DejaVu Sans')
plt.show() # show graph

Mhalo_log_lim = np.array([8, 14])
c_lim= np.array([1, 100]) # imposition of priors, originally between 100 and one 
ML_lim = np.array([0.02, 5.0]) # imposition of priors, originals betweeen 0.02 and 5
#katz et al places bounds on M* of 0.3<x<0.8

####################################################################################
#Data from Lapi 2018 https://iopscience.iop.org/article/10.3847/1538-4357/aabf35/pdf

LapilogMstar = [7.89,7.97,8.25,8.15,8.34,7.77,8.49,8.41,8.27,8.46]
LapilogVopt = [1.84,1.88,1.83,1.89,1.86,1.61,1.84,1.81,1.79,1.82]

####################################################################################

logMhalo = []
logMgal = []
logConc = []
logLum = []
rad2_2 = [] # array to list galaxy cicular velocity at 2.2R
gal2_2 = []

#galaxy_name = "CamB" #comment out for no galaxy name needed

def HMF(z , bins = 0.05 ,cum_var = 1.0):

    #Produces a Halo Mass Function (HMF) using a Tinker et al. (2008) HMF and a correction for subhalos from Behroozi et al. (2013)

    M = np.arange(9. , 15.5+bins ,bins) #Sets HMF mass range
    phi = mass_function.massFunction((10.0**M)*cosmo.h, z , mdef = '200m' , model = 'tinker08' , q_out="dndlnM") * np.log(10) * cosmo.h**3.0 * cum_var #Produces the Tinker et al. (2008) HMF

    a = 1./(1.+z) #Constants for Behroozi et al. (2013) Appendix G, eqs. G6, G7 and G8 subhalos correction
    C = np.power(10., -2.415 + 11.68*a - 28.88*a**2 + 29.33*a**3 - 10.56*a**4)
    logMcutoff = 10.94 + 8.34*a - 0.36*a**2 - 5.08*a**3 + 0.75*a**4
    correction = phi * C * (logMcutoff - M)

    return M , np.log10(phi + correction)

###################################Testing

#H_masses , H_phi = HMF(0)

#plt.figure(dpi=120,figsize = (12,12))
#plt.plot(H_masses,H_phi)
#plt.xlabel('log(M/M$_{\odot})$')
#plt.ylabel('log($\Phi$)')
#plt.show()
#print("done!")
###################################

def writematrix(SMF , sigma , z_range , title):

    #Writes a SMHM matrix to produce a SMHM for any redshift within z_range

    M_s , phi_s = SMF[0,:] , SMF[1,:]
    abun = np.loadtxt("SMHM data/Baldry + Bernardi SMHM.txt") #Reads SMHM produced from abundance matching at z = 0.1
    matrix = np.empty((0,M_s.size-1))

    if type(sigma) == float:
        #Adaptation for the code to use constant and variable values of sigma
        sigma = np.repeat(sigma,z_range.size)

    for i in range(z_range.size):
        #Calculates SMHM for each redshift in z_range
        z = z_range[i]
        sig_0 = sigma[i]
        M_h , phi_h = HMF(z)
        deri = derivative(abun[:,0],abun[:,1])(M_h)
        n=0
        e=1.

        while n < 3:
            #Calls the integrals function and matches integral values of SMF to interpolated values in the integral of HMF, iterating multiple times
            I_phi_s , I_phi_h , M_s_temp , M_h_temp = integrals(np.array([M_s , phi_s]) , np.array([M_h , phi_h]) , sig_0/deri)
            int_I_phi_h = interp1d(I_phi_h , M_h_temp , fill_value="extrapolate")
            M_h_match = np.array([])
            for m in range(M_s_temp.size):
                M_h_match = np.append(M_h_match , int_I_phi_h(I_phi_s[m]))

            M_s_iter , phi_s_iter = SMFfromSMHM(M_s_temp , M_h_match , sig_0 , z) #Reconstructing SMF
            int_phi_s_iter = interp1d(M_s_iter , phi_s_iter , fill_value="extrapolate")
            e_temp = max((phi_s - int_phi_s_iter(M_s))/phi_s) #Calculating relative error between reconstructed SMF and the input SMF

            if e_temp < e:
                #Only accepts iterations that decrease the relative error
                e = e_temp
                deri = derivative(M_h_match,M_s_temp)(M_h)
            n += 1

        matrix = np.append(matrix,[M_h_match], axis=0) #Appending each halo mass array from the SMHM at each redshift to the matrix array
    matrix = np.append(matrix,[M_s_temp], axis=0) # Appending the stellar mass array to the end of the matrix array

    with open(title+".txt" ,"w+") as file:
        #Writes the matrix to a file
        np.savetxt(file,np.array([z_range.size+1,M_s.size+1])[None],fmt="%i") #Writing the rows and columns of the file
        np.savetxt(file,np.append(z_range,np.inf)[None],fmt = "%f") #Writing the redshift of each SMHM
        np.savetxt(file,matrix.T) #Writing the SMHM matrix

##########################Writing Matrixes 

##########################

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

def r_list(galaxy):
    r_list=np.array([])
    for i in range(len(data_lines)):
        temp2 = data_lines[i]
        temp2 = temp2.split()
        if str(temp2[0]) == (galaxy):
            r_list = np.append([r_list],float(temp2[2]))
    return r_list

def Hao_AM(Halos,Galaxies,show=True,save=False,error = 0.3):
	sigma2 = error*2
	AM_HaoFu2020_galaxies= np.asarray(Galaxies)
	AM_HaoFu2020_Halos = np.asarray(Halos)
	plt.plot(AM_HaoFu2020_Halos,AM_HaoFu2020_galaxies,label='Hao Fu 2020')
	plt.title('Hao Fu Abundance Matching')
	plt.ylabel('$log_{10}[M_{*}/M_{\odot}]$')
	plt.fill_between(AM_HaoFu2020_Halos,AM_HaoFu2020_galaxies+error,AM_HaoFu2020_galaxies-error,alpha=0.5)
	plt.fill_between(AM_HaoFu2020_Halos,AM_HaoFu2020_galaxies+sigma2,AM_HaoFu2020_galaxies-sigma2,alpha=0.5)
	plt.xlabel('$log_{10}[M_{Halo}/M_{\odot}]$')
	if save == True:
		plt.savefig('SPARC DATASET-AM-'+model+"_error_"+'point2dex'+'_bounded')
	if show == True:
		plt.show()

def Rvir_calc(Mhalo_log):   #same as R200
    z = redshift # Line 241 from ForAnaCamila
    omega_m = 0.3 # Line 242 from ForAnaCamila
    HH = 0.1 * h * math.sqrt((omega_m * ((1+z)**3)) + (1-omega_m)) # Line 410,411 from ForAnaCamila
    rho_c = (3 * (HH**2)) / (8 * math.pi * G) # Line 413 from ForAnaCamila
    k = (4*math.pi) / 3 # Line 414 from ForAnaCamila
    Rvir_result = np.cbrt(((10 ** Mhalo_log) / (rho_c * 200 * k))) # Check order of operations
    return Rvir_result

def create_bins(Halo,Galaxy,binsize,show=True,save=False,plot=True): # to speed up the code state "plot=False"
    temp1, temp2 = zip(*sorted(zip(Halo, Galaxy)))
    binwidth = (temp1[-1]-temp1[0])/binsize # find width of each bin taking diff between 0th and nth element of halo array
    # then dividing by binsize
    temp3 = []
    temp6 = []
    for j in range(0,binsize): # for each bin create an empty array to append to 
        temp4 = []
        temp5 = []
        for i in range(len(temp1)): #iterate through indexes of halo array
            if temp1[0]+j*binwidth <= temp1[i] <= temp1[0]+(j+1)*binwidth: #if 0th item +bin number*bindwidth is less than halo mass 
                temp4.append(temp1[i])  #and halo mass is less than 0th item + (bin number +1) multiplied by binwidth
                temp5.append(temp2[i])
        temp3.append(sum(temp4)/len(temp4))
        temp6.append(sum(temp5)/len(temp5))
    return temp3,temp6 #halo, galaxy bins respectively

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
    X = np.log10(ML*Lum(galaxy_name)) - Mhalo_log # Subtraction as the masses are under logarithm.
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

def Mdm_DC14_calcprime(r_input,Mhalo_log,c,ML): # Calculation for dark matter halo mass under the DC14 model
    Mdm_result = np.array([])
    Rvir= Rvir_calc(Mhalo_log)
    # DC14 calculations from Di Cinto et al 2014
    X = np.log10(ML*Lum(galaxy_name)) - Mhalo_log # Subtraction as the masses are under logarithm.
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
    i=0 # The calculation must be iterated as the quad() function does not like
                                  # the use of np.array. This is inefficient and could be improved if quad()
                                  # is not used.
    r = r_input
        
    temp1 = quad(lambda x: ((x**2) / (((x / r_s)**gamma) * (1+(x / r_s)**alpha)**((beta - gamma) / alpha) )), 0, r)
    a = np.append(Mdm_result, [np.log10(4 * math.pi * Ps * temp1[0])]) # a has to be used as np.append creates a new appended array
    Mdm_result = a
    """if np.any(np.isnan(Mdm_result))==True:
    print(ML, Mhalo_log, X)"""
    
    return Mdm_result

def Vdm_calc(r_input,Mhalo_log,c, ML):

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
    return Vdm

def Vdm_calcprime(r_input,Mhalo_log,c, ML):

    '''else: # include this else function to restrict the mhalo-c relation to a gaussian distribution for NFW and DC14
        alpha = 10.84
        gamma = 0.085
        M_param = 5.5e17
        theory_mhalo_units = 10**Mhalo_log*1e-12*h
        c = 10**(np.log10((alpha *(1/theory_mhalo_units)**gamma)*(1+(theory_mhalo_units/(M_param*1e-12))**0.4))+wdm_dist)'''
    if model in ("NFW","WDM0","WDM1","WDM2","WDM3"):
        Mdm = Mdm_NFW_calc(r_input,Mhalo_log,c)
    elif model == 'DC14':
        Mdm = Mdm_DC14_calcprime(r_input,Mhalo_log,c,ML)
    else:
        Mdm = 0
        print('Error: model not known. Please select one specified in the list')
    Vdm = np.sqrt((G * 10**Mdm ) / r_input) # Equation 1, H. Katz et al. 2017
    return Vdm

	
def Moster(theory_mhalo, y):
    norm_factor = 0.0351 - 0.0247*(redshift/(redshift+1.))
    char_mass = 11.590 + 1.195*(redshift/(redshift+1.))
    beta = 1.376 - 0.826*(redshift/(redshift+1.))
    gamma = 0.608 + 0.329*(redshift/(redshift+1.))
    theory_mstar = np.log10(2 * norm_factor * 10.**theory_mhalo * ((10.**theory_mhalo/10.**char_mass)**(-beta) + (10.**theory_mhalo/10**char_mass)**gamma)**-1)
    return theory_mstar - y

def Vdm(Mdm,r):
	Vdm = np.sqrt((G * 10**Mdm ) / r)
	return Vdm # Equation 1, H. Katz et al. 2017

def Vcirc_Tot(Vdm,Vgas,ML,Vstar): # Equation to calculate the total circular velocity of a galaxy
	V_c = np.sqrt(Vdm**2+Vgas**2+(ML)*Vstar**2)
	return V_c

def logReff(logMbar): # takes logged Mbar and outputs logreff
    
    logreff = [(0.406*x-3.456) for x in logMbar]
    
    return logreff

def logRd(logreff): #takes log reff and outputs logRd
    reff = [10**x for x in logreff]
    rd = reff/1.68
    logrd = [np.log10(x) for x in rd]
    return logrd

def R(logrd): #takes logRd outputs R
    rd = [10**x for x in logrd]
    R = [2.2*x for x in rd]
    return R
    

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

def wholesample(galdata,save=False):
    return [y[0] for y in [z.split() for z in [x for x in galdata]]]
    #prints every galaxy name in sample, first finds all lines in specified galdata
    #then splits the rows into collumns, and takes the 0th item in each row and 
    #assigns them to a list

def c_NFW(Mhalo_log): # conc in NFW model
    alpha = 10.84
    gamma = 0.085
    M_param = 5.5e17
    theory_mhalo_units = 10**Mhalo_log*1e-12*h
    c = 10**(np.log10((alpha *(1/theory_mhalo_units)**gamma)*(1+(theory_mhalo_units/(M_param*1e-12))**0.4))+wdm_dist)
    return c

def X(Mstar,Mhalo):
    X = np.log10(Mstar/Mhalo)
    return X

def Mhalo_lssthn_R_NFW(Mhalo,r,c):

    M_lessthn = Mhalo*((math.log((rs+r)/rs)-(r/rs+r))/(np.log(1+c)-(c/(1+c))))

    return M_lessthn

def scaleRadius(r_vir,c):
    Rs = r_vir/c
    return Rs


def c_DC14(cNFW,Mhalo_log):
    #X = log10(Mstar/Mhalo)
    X = np.log10(ML*Lum(galaxy_name)*(10**9)) - Mhalo_log # DC14 calculations from Di Cinto et al 2014
    c = cNFW*(1.0+0.00003*exp((3.4*3.4*(X+4.5))))
    return c


def Mdm_NFW_calc_alter(r_input, Mhalo_log, c):# Calculation for dark matter halo mass under the NFW model
    r = r_input
    Rvir = Rvir_calc(Mhalo_log)
    Rs = Rvir / c
    x = r / Rs # Line 488 From ForAnaCamila. Check whether r is under logarithm

    gx = np.log(1+x) -  (x / (1 + x))
    gc = np.log(1+c) - (c / (1 + c))
    Mdm_result = Mhalo_log + np.log10(gx) - np.log10(gc) # Line 490 from ForAnaCamila
    return Mdm_result

def Mgas(logMstar): # determines Mgas From Mgal https://arxiv.org/pdf/1505.04819.pdf
    if logMstar>8.6:
        slope = 0.461
        const = 5.329

        logMgas = slope*logMstar + const # straight line equation

    if logMstar<8.6:
        slope = 1.052
        const = 0.236

        logMgas = slope*logMstar + const # straight line equation

    return logMgas

def Vstar_calc(r_list, Mstar_log): # Velocity of the stellar disk if not observed
    Vstar = np.array([]) # as arguments takes a range of radii for a specific Mstar
    for i in range(len(r_list)):
        r = r_list[i]
        a = r / Rd # Using a as a temporary variable
        a_s = a * 0.5 # a_s is half of a
        B = ( iv(a_s, 1e-15) * kv(a_s,1e-15) ) - ( iv(a_s,1) * kv(a_s,1) ) # iv and kv are modified Bessel functions
        V = math.sqrt( (0.5 * G * a * a * B * (10 ** Mstar_log)) / Rd) # Unsure about the total function in line 455 of ForAnaCamila
        a = np.append(Vstar,[V])
        Vstar = a
    return Vstar

def Vstar_calc_prime(r_list, Mstar_log): # Velocity of the stellar disk if not observed
    
    r = r_list
    a = r / Rd # Using a as a temporary variable
    a_s = a * 0.5 # a_s is half of a
    B = ( iv(a_s, 1e-15) * kv(a_s,1e-15) ) - ( iv(a_s,1) * kv(a_s,1) ) # iv and kv are modified Bessel functions
    V = math.sqrt( (0.5 * G * a * a * B * (10 ** Mstar_log)) / Rd) # Unsure about the total function in line 455 of ForAnaCamila
    a = V
    Vstar = a
    return Vstar

def Vstar_calc_alt(r_list, Mstar_log): # Velocity of the stellar disk if not observed
    Vstar = np.array([]) # as arguments takes a range of radii for a specific Mstar
    for i in range(len(r_list)):
        r = r_list[i]
        a = r # Using a as a temporary variable
        a_s = a * 0.5 # a_s is half of a
        B = ( iv(a_s, 1e-15) * kv(a_s,1e-15) ) - ( iv(a_s,1) * kv(a_s,1) ) # iv and kv are modified Bessel functions
        V = math.sqrt( (0.5 * G * a * a * B * (10 ** Mstar_log))) # Unsure about the total function in line 455 of ForAnaCamila
        a = np.append(Vstar,[V])
        Vstar = a
    return Vstar

def logReff(logMbar): # Determines log(effective radius) for a log(baryonic mass) https://arxiv.org/pdf/1505.04819.pdf
    slope = 0.406
    const = -3.456

    logReff = logMbar*slope + const

    return logReff

def Mbar(logMgal,logMgas):
    return np.log10(10**logMgal+10**logMgas)

def Radii(Reff):
    Rad = 2.2*(Reff/1.68)
    return Rad

"""
def Vgas(R,logMgas):
    V = []
    R = R/1.68
    Vgas = np.sqrt(G*10**logMgas/R)
    V.append(Vgas)
    
    for i in R:
        Vgas = np.sqrt(G*10**logMgas/R)
        V.append(Vgas)
    return Vgas
"""
def Vgas2(R,logMass):
    V = []
    for i in R:
        temp = np.sqrt((G*10**logMgas)/(i/1.68))
        V.append(temp)
    return V

def Vtot(Vdm, Vstar, ML,Vgas):
    Vtotal = np.sqrt(Vdm**2 + (ML * (Vstar**2))+Vgas**2)
    return Vtotal

def Vtot_lwrbnd(Vdm, Vstar, ML):
    Vtotal = np.sqrt(Vdm**2 + (ML * (Vstar**2)))
    return Vtotal

def klypin_VcircMvir(Mvir): #from klypin et al 2011, reference of 
    #shankar and brook. Power law for distinct haloes.
    Vcirc = 2.8*10**(-2)*Mvir**(0.316)
    return Vcirc


def Dutton2010(logMstar):
    logVmax = []
    for i in logMstar:
        
        logVmax_temp = 0.235*(i) +2.185
        logVmax.append(logVmax_temp)
    return logVmax

def pizagno2007(logMstar):
    logV2_2 = []
    for i in logMstar:
        logV2_2_temp = 2.143 + (0.281)*i #*0.7 for h units in paper
        logV2_2.append(logV2_2_temp)
    return logV2_2

def logMbar_from_logMstar(logMstar): # From Bradford paper "a study in blue"
    lessthn = []
    grtrthn = []
    for x in logMstar: #figure 5 left
        if x < 8.6:
            lessthn.append(x)
        if x > 8.6:
            grtrthn.append(x)

    #add errors in later err = 0.285
    m = 1.052
    c = 0.236
    lessthn = [(m*x + c) for x in lessthn]
    #add errors in later
    m = 0.461
    c = 5.329
    grtrthn = [(m*x + c) for x in grtrthn]
    
    logMgas = np.append(lessthn,grtrthn)
    Mgas = [10**x for x in logMgas]
    Mstar = [10**x for x in logMstar]
    
    Mbar = [(Mgas[x]+Mstar[x]) for x in range(len(Mgas))]
    logMbar = [np.log10(x) for x in Mbar]
    
    return logMbar

def H1Mass(galaxy): # needs components from dm_code-Ben-Fraser.py to function
    for i in range(len(data_lines2)): # in units of 10^9 Msol
        temp2 = data_lines2[i]
        temp2 = temp2.split()
        if str(temp2[0]) == (galaxy):
            H1 = float(temp2[13])*10e9 #now converted to units of sol mass
    return H1

def wholesample(galdata,save=False):
    return [y[0] for y in [z.split() for z in [x for x in galdata]]]
    #prints every galaxy name in sample, first finds all lines in specified galdata
    #then splits the rows into collumns, and takes the 0th item in each row and 
    #assigns them to a list

def Dutton2010prime(x,intercept,gradient,h=0.7):
    logV = intercept +gradient*(np.log10(h**2*(x/10.0**10.0)))
    return logV

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

def Lum(galaxy): # in units of 10^9 Lum Sol
    for i in range(len(data_lines2)):
        temp2 = data_lines2[i]
        temp2 = temp2.split()
        if str(temp2[0]) == (galaxy):
            lum = float(temp2[7])*10e9 # convert to units of Sol Lum
    return lum

def rad_accel(Ydisk,Ybul,Vdisk,Vbul,Vgas,R):
    gbar = Ydisk*Vdisk**2 + Ybul*Vbul**2 + Vgas**2
    gbar = gbar/R
    return gbar

def c_NFW_DC14(cNFW,Mstar,Mhalo):
    X = np.log10(Mstar/Mhalo) + 4.5
    cDC14 = cNFW*(1.0+exp(0.00001*(3.4*(X+4.5))))
    return cDC14

def logreff(x): # from bradford "a study in blue"
    #takes logged values of Mbar outputs logged values of effective radius
    slope = 0.406
    const = -3.456
    return (slope*x + const)
    
def straighline(x,m,c):
    return (m*x+c)

def c_redshift(c,z):
    c_new =  c/(1.0+z) 
    return c_new

def Mgascal(x): # from "a study in blue" by Bradford, takes in logged values
    if 10.0**x<10.0**(8.6):
        slope = 1.052
        const = 0.236
    if 10.0**x>10.0**(8.6):
        slope = 0.461
        const = 5.329
        
    return 10.0**(slope*x + const) # returns gas value

def opline(galaxy, plot = False, save = False,show=False,bounds=True,rad22=False):
    global wdm_dist
    wdm_dist = np.random.normal(0,0.16)
    #Error Checking########################

    #######################################
    ML_lim[1] = min(((Vtot_data(galaxy)+Verr(galaxy))**2 - np.sign(Vgas(galaxy))*(Vgas(galaxy)**2))/(Vstar(galaxy))**2)
    global halo_dist
    halo_dist = np.random.normal(0,0.16)
    #if model == 'NFW' or model == 'DC14':
    popt, pcov = op.curve_fit(Vtot_calc, r_list(galaxy), Vtot_data(galaxy), sigma = Verr(galaxy), p0 = (11, 1.0, ML_lim[1]), bounds = ([Mhalo_log_lim[0],c_lim[0],ML_lim[0]],[Mhalo_log_lim[1],c_lim[1],ML_lim[1]]), max_nfev=20000, method = 'trf')
    #elif model in ('WDM1', 'WDM2', 'WDM3'):
        #popt, pcov = op.curve_fit(opM, r_list(galaxy), Vtot_data(galaxy), sigma = Verr(galaxy), bounds = ([wdmh,c_lim[0],ML_lim[0]],[wdmh+0.00001,c_lim[1],ML_lim[1]]), max_nfev=10000, method = 'trf')
    #if model in ('WDM1', 'WDM2', 'WDM3'): # change this if statement to all models to restrict the mhalo-c relation to a gaussian distribution for NFW and DC14
    logMhalo.append(popt[0])
    logMgal.append(np.log10(popt[2]*(Lum(galaxy))))
    logConc.append(np.log10(popt[1])) 
    logLum.append(np.log10(Lum(galaxy)))

    #popt[1] = wdmc #I have no clue what this line does 7/2/2021
    """
    popt[0] = g_Mhalo # leave these 2 lines in to include the Mhalo-Mstar gaussian dist
    popt[2] = g_ML    # comment out to leave it unrestricted
    """
    # popt is an array of the optimal values for logMhalo, c and ML
    #pcov is the covarience matrix of Mhalo_log,c and ML
    # The argument in the op.curve_fit "bounds = ([],[])" are the bounds for the optimised values
    # The first array is the lower bounds, the second array is the upper bounds which I have left as +infinity
    #print(popt)
    if rad22 == True:
        interpolation = interp1d(r_list(galaxy), Vtot_data(galaxy))
        rad2_2.append(interpolation(2.2)) # obtain velocity at 2.2R
        gal2_2.append(Vstar(galaxy)) #obtain M* to overplot sparc sample
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
            if bounds==True:
                plt.savefig(model+galaxy+'bounds')

            if show == True:
                plt.show()
############################# HYPERPARAMETERS
redshift = 0.0

"Choose model; NFW or DC14 or WDM"
# Write the name of the model here. Ensure that it is written exactly as in the list
model = "NFW"
redshift = 0.0 #currently for 0.0 or 1.1

wdm_dist = 0
halo_dist = 0
g_ML = 0
g_Mhalo = 0
Rd = 1.68
al_gas = 1
Ms = 5
# Ms is the input free-streaming  scale for the WDM model

#h = 0.7

G = 4.302*(10**(-6)) # The gravitational constant in Kpc/Msun(km/s)^2

ML = 1.111308962394226

dist = False # True to include the Mhalo-Mstar gaussian dist, false to leave it unrestricted
############################# SHMs with scatter
#masses_scatter , phis = SMFfromSMHM(AM_HaoFu2020_galaxies, AM_HaoFu2020_Halos, 0.2, 0)


#############################
#Velocity function

Mvir = [10**x for x in AM_HaoFu2020_galaxies] # this might not be virial mass, check!!!
Vvir = [klypin_VcircMvir(x) for x in Mvir]

logMvir = [np.log10(x) for x in Mvir]
logVvir = [np.log10(x) for x in Vvir]

plt.figure(figsize=(12,12),dpi=60)
plt.plot(logMvir,logVvir) # Velocity function
plt.xlabel('logM$_{vir}$')
plt.ylabel('logV$_{vir}$')
plt.plot()
plt.show()

Mvir = [10**x for x in AM_HaoFu2020_galaxies] # this might not be virial mass, check!!!
Vvir = [[klypin_VcircMvir(x) for x in Mvir][i]/(Mvir[i])**(1./3.) for i in range(len(Vvir))]
logMvir = [np.log10(x) for x in Mvir]
logVvir = [np.log10(x) for x in Vvir]

plt.figure(figsize=(12,12),dpi=60)
plt.plot(logMvir,logVvir) # Velocity function
plt.xlabel('log(M$_{vir}$)')
plt.ylabel('log(V$_{vir}$/M$^{1/3}$) (km s$^{-1}$(h$^{-1}$M$_{\odot}$)$^{1/3}$)')
plt.plot()
plt.show()
########################
"""
f = open('tab_3kev.dat',"r") #File name of concentration data
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
plt.xlabel("theory_mhalo")
plt.ylabel("theory c")
plt.show()
"""
#print(wholesample(data_lines1))

#Hao_AM(AM_HaoFu2020_Halos,AM_HaoFu2020_galaxies,show=False)
#

Thry_Halo,Thry_Gal = create_bins(AM_HaoFu2020_Halos,AM_HaoFu2020_galaxies,50)#final number is number of bins to plot
Bins = [[Thry_Halo[x],Thry_Gal[x]] for x in range(len(Thry_Halo))]

plt.plot(Thry_Halo,Thry_Gal)

##################################### #Add Mgas, Mbar, Reff, and Rd to the binned arrays

for i in range(len(Bins)): # iterate over all bins
    #Bins:
    #Bins[0]:MHalo
    #Bins[1]:Mgalaxy
    #Bins[2]:Mgas
    #Bins[3]:Mbar
    #Bins[4]:Reff
    #Bins[5]:Rd
    plt.scatter(Bins[i][0],Bins[i][1]) #Mgal vs Mhalo
    Bins[i].append(Mgas(Bins[i][1])) # Calculating log Mgas from log Mgal
    Bins[i].append(Mbar(Bins[i][1],Bins[i][2])) #Calculating  
    Bins[i].append(logReff(Bins[i][3])) # Calculates logReff
    Bins[i].append(Bins[i][4]/1.68)
    
    print("logMhalo:",Bins[i][0],"logMgal: ",Bins[i][1],"logMgas:",Mgas(Bins[i][2]),
        "logMbar:",Bins[i][3],"Reff:",Bins[i][4],"Reff:",Bins[i][5])
#lohMbaryonic will be put in Bin[2]
#lohMbaryonic= Mgal+Mgas
#logReff will be put in Bin[3]
#log Reff has relation to Mbar given in function logReff


##################################### #Plot Mgal vs Mhalo

plt.xlabel('log(M$_{Halo}$/M$_{\odot}$)')
plt.ylabel('log(M$_{Galaxy}$/M$_{\odot}$)')
plt.show()

##################################### #Plot Mas vs Mgal

for i in range(len(Bins)): #Mgas vs MGal # iterate over all bins
    plt.scatter(Bins[i][1],Bins[i][2])

plt.xlabel('log(M$_{Galaxy}$ [M$_{\odot}$])')
plt.ylabel('log(M$_{Gas}$) [kpc])')
plt.show()

##################################### Plot Reff vs Mbar

for i in range(len(Bins)): # iterate over all bins
    plt.scatter(Bins[i][3],Bins[i][4]) #Reff vs Mbar

plt.xlabel('log(M$_{Baryon}$ [M$_{\odot}$])')
plt.ylabel('log(r$_{eff}$) [kpc])')
plt.show()

#############################
#Moster vs hao testing

halos = [ np.log10(x) for x in np.logspace(10, 16)] # create synthetic halos for use in moster function
synth_gals = [Moster(x,0) for x in halos]

print(synth_gals)
plt.figure(figsize=(12,12),dpi=60)
plt.xlabel('log(M$_{Halo}$) [M$_{\odot}$]')
plt.ylabel('log(M$_{Gal}$) [M$_{\odot}$]')
plt.plot(Thry_Halo,Thry_Gal,label='Hao 2020')
plt.plot(halos,synth_gals,label='Moster')
plt.legend()
plt.show()


############################################################# Velocity Calculation

Modot = 6.0e24

############################################################
#calc Velocities at 2.2R for all SPARC sample
############################################################

logLums = [np.log10(z) for z in [Lum(x) for x in wholesample(data_lines1)]] # repeating Lelli Calculations
logMassesH1 = [np.log10(z) for z in [H1Mass(x) for x in [y for y in wholesample(data_lines1)]]] #2 lines produce logged luminosities
#and logged masses of H1 for plotting in reference to SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry and 
#Accurate Rotation Curves https://ui.adsabs.harvard.edu/abs/2016AJ....152..157L/abstract

plt.figure(dpi=100,figsize=(12,12))
plt.scatter(logLums,logMassesH1)

plt.show()


#for i in wholesample(data_lines1):
#    galaxy_name = i
    

#    opline(i)
galaxy_name = "CamB"

##########################################################
#Calc Velocities at 2.2R in theory only 
##########################################################
scale_length = 2.2
redshift = 0.0

#Give range of Mbar
Mhalos = np.linspace(8,14,1000) # create pre-logged halos
unloggedhalos = [10**x for x in Mhalos]
Mgals = [10**Moster(x,0) for x in Mhalos]# create range of galaxy masses unlogged
logMgals = [np.log10(x) for x in Mgals]#*h**2
#Mgals = np.logspace(6,12) 
Mbaryonic =  [x for x in Mgals]#Mgas not being included as looking at lower bound
Mgascalc = [Mgascal(np.log10(x)) for x in Mgals] # Mgascal takes logged values of M* returns unlogged values of Mgas
Mbaryonicplus = [np.log10(Mbaryonic[i] + Mgascalc[i]) for i in range(len(Mgals))]
#print("here",Mbaryonicplus)

plt.plot(np.log10(Mgals),Mbaryonicplus)
plt.show()

logreffs = [logreff(x) for x in Mbaryonicplus]

plt.plot(Mbaryonicplus,logreffs)
plt.show()

scale_radii = [(10**x)/1.68 for x in logreffs] # create scale radii for each galaxy 
print(scale_radii)


#Vc**2 = Vhalo**2 + Vgal**2 + Vbary**2

#R = 2.2*Rd

radius = [2.2*x for x in scale_radii]


model='NFW'
cNFWtemp = [c_NFW(x) for x in Mhalos]
halovelocityNFW = [Vdm_calc(radius[x], Mhalos[x], cNFWtemp[x], ML) for x in range(len(Mgals))]
model='DC14'
cDC14temp = [c_NFW_DC14(c_NFW(Mhalos[x]),Mgals[x],[10**x for x in Mhalos][x]) for x in range(len(Mhalos))]
#print(cNFWtemp,cDC14temp)
halovelocityDC14 = [Vdm_calcprime(radius[x], Mhalos[x], cDC14temp[x], ML) for x in range(len(Mgals))]

BaryonicVel = [Vstar_calc_prime(radius[x], [np.log10(y) for y in Mbaryonic][x]) for x in range(len(Mhalos))] 


plt.plot(Mhalos,halovelocityNFW,label="NFW")
plt.plot(Mhalos,halovelocityDC14,label="DC14")
plt.legend()
plt.show()


totalvelocitysumNFW = [np.log10(np.sqrt(halovelocityNFW[x]**2.0+BaryonicVel[x]**2.0)) for x in range(len(Mhalos))]
totalvelocitysumDC14 = [np.log10(np.sqrt(halovelocityDC14[x]**2.0+BaryonicVel[x]**2.0)) for x in range(len(Mhalos))]
#print("look here",halovelocityNFW[0],halovelocityDC14[0],BaryonicVel[0])

Dutton10 = [Dutton2010prime(x, 2.185, 0.235) for x in Mgals]
Pizagno07 = [Dutton2010prime(x, 2.143, 0.281) for x in Mgals]
#Pizagno07 = [straighline(x,0.281,2.143) for x in logMgalsh2]
xaxis = [np.log10(x) for x in Mgals]
plt.figure(dpi=120,figsize=(8,8))
#plt.xlim((9,11))
plt.plot([np.log10(x) for x in Mgals],totalvelocitysumNFW,label="NFW")
plt.plot([np.log10(x) for x in Mgals],totalvelocitysumDC14,label="DC14")
plt.plot(xaxis,Dutton10,label="Dutton 10")
plt.plot(xaxis,Pizagno07,label="Pizagno 07")
plt.title('Log(V$_C$/[kms$^{-1}$]) against log(M$_{Gal}$/M$_\odot$)')
plt.xlabel('Log(M$_{Gal}$/M$_\odot$)')
plt.ylabel("Log(V$_C$)")
plt.legend()
plt.show()

#Ideas
#Replace Moster with Bernardi-Baldry Relation and observe difference


################################################################################
#Create Vel Curve for Acc Relation
################################################################################

Mhalos = np.linspace(8,14) # create pre-logged halos
unloggedhalos = [10**x for x in Mhalos]
Mgals = [10**Moster(x,0) for x in Mhalos]# create range of galaxy masses unlogged
logMgals = [np.log10(x) for x in Mgals]#*h**2
Mbaryonic =  [x for x in Mgals]#Mgas not being included as looking at lower bound
Mgascalc = [Mgascal(np.log10(x)) for x in Mgals] # Mgascal takes logged values of M* returns unlogged values of Mgas
Mbaryonicplus = [np.log10(Mbaryonic[i] + Mgascalc[i]) for i in range(len(Mgals))]

radiusprime = np.arange(0.1, 30.1, 1.0)
redshift = 0.0

calc_radius = 10.0

for i in range(len(Mhalos)): 
    Vstar_array = np.array(Vstar_calc(radiusprime, logMgals[i])) # testwith swapping log Mstar with log Mgas
                                                                #dark matter velocity ends up way too high
    
    
    # Turns Vdm into an array to be plotted
    V_dm_array = np.array(Vdm_calc(radiusprime, Mhalos[i], c_NFW(Mhalos[i]),ML))
    
    
    # Total velocity values to be plotted
    Total_Velocity = np.array(Vtot_lwrbnd(V_dm_array, Vstar_array, ML)) # lower bound of velocity
    
    plt.plot(radiusprime,[np.log10(x) for x in[1.0e3*Total_Velocity[x]/(3.086e19*radiusprime[x]) for x in range(len(Total_Velocity))]]) #convert kpc to m
plt.show()
    
    
    
    
################################################################################
#Acceleration relations
################################################################################
#Hopkins Vc/r = g_obs


Vc_NFW = totalvelocitysumNFW
Vc_DC14 = totalvelocitysumDC14

g_obs_NFW = [Vc_NFW[x]/(radius[x]) for x in range(len(Vc_NFW))]
g_obs_DC14 = [Vc_DC14[x]/(radius[x]) for x in range(len(Vc_DC14))]

log_g_obs_NFW = [np.log10(x) for x in g_obs_NFW]
log_g_obs_DC14 = [np.log10(x) for x in g_obs_DC14]

#gobs =sqrt(† g g bar) where ( g) † = »´ A - -- g G 1.5 10 m s 1 10 2 HOPKINS

Const = 1.5*10**(-10) #ms^-2

g_bar_NFW = [(g_obs_NFW[x]**2)/Const for x in range(len(g_obs_NFW))]
g_bar_DC14 = [(g_obs_DC14[x]**2)/Const for x in range(len(g_obs_DC14))]

log_g_bar_NFW = [np.log10(x) for x in g_bar_NFW]
log_g_bar_DC14 = [np.log10(x) for x in g_bar_DC14]

plt.plot(log_g_bar_NFW,log_g_obs_NFW)
plt.plot(log_g_bar_DC14,log_g_obs_DC14)
plt.show()


################################################################################

acc_radius = 10.0 #set radius to calculate acceleration at in kpc

Mhalos = np.linspace(8,14) # create pre-logged halos
unloggedhalos = [10**x for x in Mhalos]
Mgals = [10**Moster(x,0) for x in Mhalos]# create range of galaxy masses unlogged
logMgals = [np.log10(x) for x in Mgals]#*h**2
Mbaryonic =  [x for x in Mgals]#Mgas not being included as looking at lower bound
Mgascalc = [Mgascal(np.log10(x)) for x in Mgals] # Mgascal takes logged values of M* returns unlogged values of Mgas
Mbaryonicplus = [np.log10(Mbaryonic[i] + Mgascalc[i]) for i in range(len(Mgals))]

model='NFW'
cNFWtemp = [c_NFW(x) for x in Mhalos]
halovelocityNFW = [Vdm_calc(acc_radius, Mhalos[x], cNFWtemp[x], ML) for x in range(len(Mgals))]


model='DC14'
cDC14temp = [c_NFW_DC14(c_NFW(Mhalos[x]),Mgals[x],[10**x for x in Mhalos][x]) for x in range(len(Mhalos))]
halovelocityDC14 = [Vdm_calcprime(acc_radius, Mhalos[x], cDC14temp[x], ML) for x in range(len(Mgals))]


BaryonicVel = [Vstar_calc_prime(acc_radius, [np.log10(y) for y in Mbaryonic][x]) for x in range(len(Mhalos))] 


totalvelocitysumNFW = [np.sqrt(halovelocityNFW[x]**2.0+BaryonicVel[x]**2.0) for x in range(len(Mhalos))]
totalvelocitysumDC14 = [np.sqrt(halovelocityDC14[x]**2.0+BaryonicVel[x]**2.0) for x in range(len(Mhalos))]


g_obs_NFW_p = [((1e3*totalvelocitysumNFW[x])**2.0)/(3.086e19*acc_radius) for x in range(len(Mhalos))]
g_obs_DC14_p = [((1e3*totalvelocitysumDC14[x])**2.0)/(3.086e19*acc_radius) for x in range(len(Mhalos))]

g_bar_p = [((1e3*BaryonicVel[x])**2.0)/(3.086e19*acc_radius) for x in range(len(BaryonicVel))]

log_g_obs_NFW_p = [np.log10(x) for x in g_obs_NFW_p]
log_g_obs_DC14_p = [np.log10(x) for x in g_obs_DC14_p]

log_g_bar_p = [np.log10(x) for x in g_bar_p]
print(log_g_bar_p)

plt.xlim(10**(-13.0),10**(-9.0))
plt.ylim(10**(-13.0),10**(-9.0))
plt.yscale("log")
plt.xscale("log")
plt.plot(g_bar_p,g_obs_NFW_p,label="NFW")
plt.plot(g_bar_p,g_obs_DC14_p,label='DC14')
#plt.show()
################################################################################
def theory_g(x, g_scale):    #function fitting char(), McGaugh 2016
    return x / (1 - np.exp(- np.sqrt(x/g_scale))) 

theory_1to1 = np.linspace(10**-12.5, 10**-9)
plt.xlim(10**(-13.0),10**(-9.0))
plt.ylim(10**(-13.0),10**(-9.0))
plt.yscale("log")
plt.xscale("log")
plt.plot(theory_1to1, theory_1to1, color = "black", linestyle='--') #No DM, 1:1 line
plt.plot(theory_1to1, theory_g(theory_1to1, 1.2e-10), label="McGaugh", color = "red")    #McGaugh fit 
plt.xlabel("g$_{Bar}$ [ms$^{-1}$]")
plt.ylabel("g$_{Obs}$ [ms$^{-1}$]")
plt.legend()
plt.show()

################################################################################
###########################################################
#No Vgas in this plot as cannot get its functional form right

if input('plot z=0?') == "Y":
    redshift = 0.0
    model="NFW" # beyond this point are NFW
    totalV = [] #create empty list for velocity curve
    galmass = []
    Mbary = [] # array for baryonic mass testing
    for i in range(len(Bins)):
        BinNumber = i
        log_Mhalo_theory = Bins[BinNumber][0] #Mhalo Bin
        log_Mstar_theory = Bins[BinNumber][1] #Mgal Bin
        log_Mbar = Bins[BinNumber][3] #Mbar bin
        # Range of radii to be plotted
        radius = np.arange(1, 30.1, 0.0025)


        # Calculates values of Vstar as an array
        Vstar_array = np.array(Vstar_calc(radius, log_Mstar_theory)) # testwith swapping log Mstar with log Mgas
                                                                    #dark matter velocity ends up way too high


        # Turns Vdm into an array to be plotted
        V_dm_array = np.array(Vdm_calc(radius, log_Mhalo_theory, c_NFW(log_Mhalo_theory),ML))


        # Total velocity values to be plotted
        Total_Velocity = np.array(Vtot_lwrbnd(V_dm_array, Vstar_array, ML)) # lower bound of velocity
        
        totvelocity = interp1d(radius, Total_Velocity) # setup interpolation for velocity curve interp(x,y)
        #totvelociy is now a function of x that can be called like y = totvelocity(x)
        TotV_NFW = totvelocity(2.2) #total velocity at R =2.2
        totalV.append(TotV_NFW)
        galmass.append(log_Mstar_theory)
        Mbary.append(log_Mbar)

        print('Mhalo:', log_Mhalo_theory)
        print('Mstar', log_Mstar_theory)
        print('Rvir:', Rvir_calc(log_Mhalo_theory))
        print('c:', c_NFW(log_Mhalo_theory))
        print('Mdm:', Mdm_NFW_calc(radius, log_Mhalo_theory, c_NFW(log_Mhalo_theory)))
        print('Vdm:', V_dm_array)
        print('Vstar:', ML*Vstar_array)
        # Plotting rotation curve

        plt.figure(figsize = (15,15),dpi=120)
        plt.plot(radius, V_dm_array, 'k-', label = 'DM Velocity')
        plt.plot(radius, np.sqrt(ML)*Vstar_array, 'r-', label = 'Baryonic Velocity')
        plt.plot(radius, Total_Velocity, 'b-', label = 'Total velocity')
        plt.xlabel("Radius - kpc")
        plt.ylabel("Velocity - km/s")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper right", borderaxespad=0.)
        plt.title('Rotation curve, log(M$_{Star}$):'+str(log_Mstar_theory)+"[M$_{\odot}$] z="+str(redshift))
        plt.show()
        
        
    totalVDC14 = [] #create empty list for velocity curve
    galmassDC14 = []
    MbaryDC14 = [] # array for baryonic mass testing
    model = "DC14" # below this point are DC14 models
    for i in range(len(Bins)):
        BinNumber = i
        log_Mhalo_theory = Bins[BinNumber][0] #Mhalo Bin
        log_Mstar_theory = Bins[BinNumber][1] #Mgal Bin
        log_Mbar = Bins[BinNumber][3] #Mbar bin
        # Range of radii to be plotted
        radius = np.arange(1, 30.1, 0.0025)


        # Calculates values of Vstar as an array
        Vstar_array = np.array(Vstar_calc(radius, log_Mstar_theory)) # testwith swapping log Mstar with log Mgas
                                                                    #dark matter velocity ends up way too high


        # Turns Vdm into an array to be plotted
        V_dm_array = np.array(Vdm_calc(radius, log_Mhalo_theory, c_NFW_DC14(c_NFW(log_Mhalo_theory), log_Mstar_theory, log_Mhalo_theory),ML))#c_DC14(c_NFW(log_Mhalo_theory),log_Mhalo_theory),ML))
        
        #print("DC14 DM:",V_dm_array)

        # Total velocity values to be plotted
        Total_Velocity = np.array(Vtot_lwrbnd(V_dm_array, Vstar_array, ML)) # lower bound of velocity
        
        totvelocity = interp1d(radius, Total_Velocity) # setup interpolation for velocity curve interp(x,y)
        #totvelociy is now a function of x that can be called like y = totvelocity(x)
        TotV_DC14 = totvelocity(2.2) #total velocity at R =2.2
        totalVDC14.append(TotV_DC14)
        galmassDC14.append(log_Mstar_theory)
        MbaryDC14.append(log_Mbar)

        print('Mhalo:', log_Mhalo_theory)
        print('Mstar', log_Mstar_theory)
        print('Rvir:', Rvir_calc(log_Mhalo_theory))
        print('c:', c_redshift(c_NFW(log_Mhalo_theory),0))
        print('Mdm:', Mdm_DC14_calc(radius, log_Mhalo_theory, c_NFW_DC14(c_NFW(log_Mhalo_theory), log_Mstar_theory, log_Mhalo_theory),ML))
        print('Vdm:', V_dm_array)
        print('Vstar:', ML*Vstar_array)
        # Plotting rotation curve

        plt.figure(figsize = (15,15),dpi=120)
        plt.plot(radius, V_dm_array, 'k-', label = 'DM Velocity')
        plt.plot(radius, np.sqrt(ML)*Vstar_array, 'r-', label = 'Baryonic Velocity')
        plt.plot(radius, Total_Velocity, 'b-', label = 'Total velocity')
        plt.xlabel("Radius - kpc")
        plt.ylabel("Velocity - km/s")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper right", borderaxespad=0.)
        plt.title('Rotation curve, log(M$_{Star}$):'+str(log_Mstar_theory)+"[M$_{\odot}$] z="+str(redshift))
        plt.show()
    
    

#IDEAS:
#POLYFIT 20th order polynomial to the Mgas Vgas relationships from SPARC at 2.2R
#Then run an interpolation to get approx values to compare to bradfords paper
#Calculate the velocities for each SPARC galaxy and overplot on different relations

#Change the x axis on velocity function graph to be baryonic mass
############################### Plot a velocity function at certain radius


    ######################## Dutton Data
    xdata = np.arange(6.5,12)
    ydata = Dutton2010(xdata)
    
    ydata_piz = pizagno2007(xdata)
    
    ########################    
    ydata = [x*h**2 for x in ydata] # adding h factor as seen in duttons paper
    ydata_piz = [x*h**2 for x in ydata_piz] # same factor change

    ########################

    totalV = [np.log10(x) for x in totalV] # log10 of velocity
    plt.figure(figsize = (12,12), dpi=60)
    plt.plot(galmass,totalV,label="NFW") # logVc vs log M* NFW
    plt.plot(galmassDC14,[np.log10(x) for x in totalVDC14],label="DC14")
    plt.plot(xdata,ydata,label="Dutton",linestyle="--") # Plot Dutton's relations
    #plt.plot(Mbary,totalV,label="logVc vs LogMbar") # logVc vs log Mbar
    plt.plot(xdata,ydata_piz,label="Pizagno 07",linestyle="--")
    #plt.scatter(gal2_2, rad2_2) # rad2_2 is the velcocity at 2.2R 
    #plt.scatter(LapilogMstar,LapilogVopt,label="Lapi 08 Vopt") # data from Lapi 2018
    
    plt.xlabel("log(M*/M$_{\odot}$)")
    plt.ylabel("log(V$_C$)")
    plt.legend()
    #plt.xlim(6,12)
    #plt.ylim(1.6,2.7)
    plt.title('Rotation curve, log(M$_{Star}$):'+str(log_Mstar_theory)+"[M$_{\odot}$] z="+str(redshift))
    plt.show()
        
    print(len(gal2_2),len(rad2_2))
    plt.figure(dpi=120,figsize=(12,12))
    plt.title("Radial Acceleration against Galaxy Mass"+", z = "+str(redshift))
    plt.xlabel("log(M$_{Gal}$/M$_\odot$)")
    plt.ylabel("g [ms$^{-1}$]")
    plt.plot(galmass,([(x**2)/2.2 for x in totalV]),label='NFW')
    plt.plot(galmass,[(x**2)/2.2 for x in [np.log10(x) for x in totalVDC14]],label="DC14")
    plt.legend()
    plt.show()



###############################################################


if input('plot z=1?') == "Y":
    redshift = 1.0
    model="NFW" # beyond this point are NFW
    totalV = [] #create empty list for velocity curve
    galmass = []
    Mbary = [] # array for baryonic mass testing
    for i in range(len(Bins)):
        BinNumber = i
        log_Mhalo_theory = Bins[BinNumber][0] #Mhalo Bin
        log_Mstar_theory = Bins[BinNumber][1] #Mgal Bin
        log_Mbar = Bins[BinNumber][3] #Mbar bin
        # Range of radii to be plotted
        radius = np.arange(1, 30.1, 0.0025)


        # Calculates values of Vstar as an array
        Vstar_array = np.array(Vstar_calc(radius, log_Mstar_theory)) # testwith swapping log Mstar with log Mgas
                                                                    #dark matter velocity ends up way too high

        
        #print("C:",c_NFW(log_Mhalo_theory))
        #print("c+:",(c_NFW(log_Mhalo_theory))/2)
        # Turns Vdm into an array to be plotted
        V_dm_array = np.array(Vdm_calc(radius, log_Mhalo_theory,c_NFW(log_Mhalo_theory)/2,ML))


        # Total velocity values to be plotted
        Total_Velocity = np.array(Vtot_lwrbnd(V_dm_array, Vstar_array, ML)) # lower bound of velocity
        
        totvelocity = interp1d(radius, Total_Velocity) # setup interpolation for velocity curve interp(x,y)
        #totvelociy is now a function of x that can be called like y = totvelocity(x)
        TotV_NFW = totvelocity(2.2) #total velocity at R =2.2
        totalV.append(TotV_NFW)
        galmass.append(log_Mstar_theory)
        Mbary.append(log_Mbar)

        print('Mhalo:', log_Mhalo_theory)
        print('Mstar', log_Mstar_theory)
        print('Rvir:', Rvir_calc(log_Mhalo_theory))
        print('c:', c_NFW(log_Mhalo_theory))
        print('Mdm:', Mdm_NFW_calc(radius, log_Mhalo_theory, c_NFW(log_Mhalo_theory)))
        print('Vdm:', V_dm_array)
        print('Vstar:', ML*Vstar_array)
        # Plotting rotation curve

        plt.figure(figsize = (15,15),dpi=120)
        plt.plot(radius, V_dm_array, 'k-', label = 'DM Velocity')
        plt.plot(radius, np.sqrt(ML)*Vstar_array, 'r-', label = 'Baryonic Velocity')
        plt.plot(radius, Total_Velocity, 'b-', label = 'Total velocity')
        plt.xlabel("Radius - kpc")
        plt.ylabel("Velocity - km/s")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper right", borderaxespad=0.)
        plt.title('Rotation curve, log(M$_{Star}$):'+str(log_Mstar_theory)+"[M$_{\odot}$] z="+str(redshift))
        plt.show()
        
        
    totalVDC14 = [] #create empty list for velocity curve
    galmassDC14 = []
    MbaryDC14 = [] # array for baryonic mass testing
    model = "DC14" # below this point are DC14 models
    for i in range(len(Bins)):
        BinNumber = i
        log_Mhalo_theory = Bins[BinNumber][0] #Mhalo Bin
        log_Mstar_theory = Bins[BinNumber][1] #Mgal Bin
        log_Mbar = Bins[BinNumber][3] #Mbar bin
        # Range of radii to be plotted
        radius = np.arange(1, 30.1, 0.0025)


        # Calculates values of Vstar as an array
        Vstar_array = np.array(Vstar_calc(radius, log_Mstar_theory)) # testwith swapping log Mstar with log Mgas
                                                                    #dark matter velocity ends up way too high


        # Turns Vdm into an array to be plotted
        V_dm_array = np.array(Vdm_calc(radius, log_Mhalo_theory, c_redshift(c_NFW_DC14(c_NFW(log_Mhalo_theory), log_Mstar_theory, log_Mhalo_theory),1),ML))#c_DC14(c_NFW(log_Mhalo_theory),log_Mhalo_theory),ML))
        
        #print("DC14 Conc manual:",c_NFW(log_Mhalo_theory)/2)
        #print("DC14 Conc", c_redshift(c_NFW(log_Mhalo_theory), 1))
        # Total velocity values to be plotted
        Total_Velocity = np.array(Vtot_lwrbnd(V_dm_array, Vstar_array, ML)) # lower bound of velocity
        
        totvelocity = interp1d(radius, Total_Velocity) # setup interpolation for velocity curve interp(x,y)
        #totvelociy is now a function of x that can be called like y = totvelocity(x)
        TotV_DC14 = totvelocity(2.2) #total velocity at R =2.2
        totalVDC14.append(TotV_DC14)
        galmassDC14.append(log_Mstar_theory)
        MbaryDC14.append(log_Mbar)

        print('Mhalo:', log_Mhalo_theory)
        print('Mstar', log_Mstar_theory)
        print('Rvir:', Rvir_calc(log_Mhalo_theory))
        print('c:', c_NFW_DC14(c_NFW(log_Mhalo_theory), log_Mstar_theory, log_Mhalo_theory))
        print('Mdm:', Mdm_DC14_calc(radius, log_Mhalo_theory, c_redshift(c_NFW_DC14(c_NFW(log_Mhalo_theory), log_Mstar_theory, log_Mhalo_theory),1),log_Mhalo_theory),ML)
        print('Vdm:', V_dm_array)
        print('Vstar:', ML*Vstar_array)
        # Plotting rotation curve

        plt.figure(figsize = (15,15),dpi=120)
        plt.plot(radius, V_dm_array, 'k-', label = 'DM Velocity')
        plt.plot(radius, np.sqrt(ML)*Vstar_array, 'r-', label = 'Baryonic Velocity')
        plt.plot(radius, Total_Velocity, 'b-', label = 'Total velocity')
        plt.xlabel("Radius - kpc")
        plt.ylabel("Velocity - km/s")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper right", borderaxespad=0.)
        
        plt.show()


#IDEAS:
#POLYFIT 20th order polynomial to the Mgas Vgas relationships from SPARC at 2.2R
#Then run an interpolation to get approx values to compare to bradfords paper
#Calculate the velocities for each SPARC galaxy and overplot on different relations

#Change the x axis on velocity function graph to be baryonic mass
############################### Plot a velocity function at certain radius


    ######################## Dutton Data
    xdata = np.arange(6.5,12)
    ydata = Dutton2010(xdata)
    
    ydata_piz = pizagno2007(xdata)
    
    ########################    
    
    ydata = [x*h**2 for x in ydata] # adding h factor as seen in duttons paper
    ydata_piz = [x*h**2 for x in ydata_piz] # same factor change

    totalV = [np.log10(x) for x in totalV] # log10 of velocity
    plt.figure(figsize = (12,12), dpi=60)
    plt.plot(galmass,totalV,label="NFW") # logVc vs log M* NFW
    plt.plot(galmassDC14,[np.log10(x) for x in totalVDC14],label="DC14")
    plt.plot(xdata,ydata,label="Dutton",linestyle="--") # Plot Dutton's relations
    #plt.plot(Mbary,totalV,label="logVc vs LogMbar") # logVc vs log Mbar
    plt.plot(xdata,ydata_piz,label="Pizagno 07",linestyle="--")
    #plt.scatter(gal2_2, rad2_2) # rad2_2 is the velcocity at 2.2R 
    #plt.scatter(LapilogMstar,LapilogVopt,label="Lapi 08 Vopt") # data from Lapi 2018
    plt.xlabel("log(M*/M$_{\odot}$)")
    plt.ylabel("log(V$_C$)")
    plt.legend()
    plt.xlim(6,12)
    plt.ylim(1.6,2.7)
    plt.show()
        
    print(len(gal2_2),len(rad2_2))
    
    plt.figure(dpi=120,figsize=(12,12))
    plt.title("Radial Acceleration against Galaxy Mass"+", z = "+str(redshift))
    plt.xlabel("log(M$_{Gal}$/M$_\odot$)")
    plt.ylabel("g [ms$^{-1}$]")
    plt.plot(galmass,([(x**2)/2.2 for x in totalV]), label='NFW')
    plt.plot(galmass,[(x**2)/2.2 for x in [np.log10(x) for x in totalVDC14]], label='DC14')
    plt.legend()
    plt.show()

"""
#############################################################
#THIS METHOD APPROXIMATES YOU CALCULATE VGAS BY THE SAME METHOD AS VSTAR
#THIS IS VERY LIKELY WRONG

if input('plot include Vgas?') == "Y":
    # Values of Mhalo and Mstar
    log_Mhalo_theory = 12          #np.arange(10, 14.1, 0.1)
    log_Mstar_theory = Moster(log_Mhalo_theory, 0) # replace moster with hao values

    BinNumber = 1
    plot_all = "True"
    if plot_all == "True":
        for i in range(len(Bins)):
            BinNumber = i
            log_Mhalo_theory = Bins[BinNumber][0] #Mhalo Bin
            log_Mstar_theory = Bins[BinNumber][1] #Mgal Bin
            log_Mgas_theory = Bins[BinNumber][2]
            # Range of radii to be plotted
            radius = np.arange(1, 30.1, 0.1)


            # Calculates values of Vstar as an array
            Vstar_array = np.array(Vstar_calc(radius, log_Mstar_theory)) # testwith swapping log Mstar with log Mgas
                                                                        #dark matter velocity ends up way too high


            # Turns Vdm into an array to be plotted
            V_dm_array = np.array(Vdm_calc(radius, log_Mhalo_theory, c_NFW(log_Mhalo_theory),ML))


            # Total velocity values to be plotted
            Total_Velocity = np.array(Vtot(V_dm_array, Vstar_array, ML,Vstar_calc(radius,log_Mgas_theory)))

            print('Mhalo:', log_Mhalo_theory)
            print('Mstar', log_Mstar_theory)
            print('Rvir:', Rvir_calc(log_Mhalo_theory))
            print('c:', c_NFW(log_Mhalo_theory))
            print('Mdm:', Mdm_NFW_calc(radius, log_Mhalo_theory, c_NFW(log_Mhalo_theory)))
            print('Vdm:', V_dm_array)
            print('Vstar:', ML*Vstar_array)
            print('Vgas:',Vstar_calc(radius,log_Mgas_theory))
            # Plotting rotation curve

            plt.figure(figsize = (15,15),dpi=120)
            plt.plot(radius, V_dm_array, 'k-', label = 'DM Velocity')
            plt.plot(radius, np.sqrt(ML)*Vstar_array, 'r-', label = 'Baryonic Velocity')
            plt.plot(radius, Vstar_calc(radius,log_Mgas_theory), label = 'Gas Velocity (est')
            plt.plot(radius, Total_Velocity, 'b-', label = 'Total velocity')


            plt.xlabel("Radius - kpc")
            plt.ylabel("Velocity - km/s")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper right", borderaxespad=0.)
            plt.title('Rotation curve, log(M$_{Star}$):'+str(log_Mstar_theory)+"[M$_{\odot}$]z="+str(redshift))
            plt.show()

###################################################
#Francesco has advised switching to not using the gas velocity therefore the Vc will be the lower bound



if input('plot z=0?') == "Y":
    totalV = [] #create empty list for velocity curve
    galmass = []
    for i in range(len(Bins)):
        BinNumber = i
        log_Mhalo_theory = Bins[BinNumber][0] #Mhalo Bin
        log_Mstar_theory = Bins[BinNumber][1] #Mgal Bin
        # Range of radii to be plotted
        radius = np.arange(1, 30.1, 0.1)


        # Calculates values of Vstar as an array
        Vstar_array = np.array(Vstar_calc(radius, log_Mstar_theory)) # testwith swapping log Mstar with log Mgas
                                                                    #dark matter velocity ends up way too high


        # Turns Vdm into an array to be plotted
        V_dm_array = np.array(Vdm_calc(radius, log_Mhalo_theory, c_NFW(log_Mhalo_theory),ML))


        # Total velocity values to be plotted
        Total_Velocity = np.array(Vtot_lwrbnd(V_dm_array, Vstar_array, ML)) # lower bound of velocity
        
        totvelocity = interp1d(radius, Total_Velocity) # setup interpolation for velocity curve interp(x,y)
        #totvelociy is now a function of x that can be called like y = totvelocity(x)
        TotV = totvelocity(2.2)
        totalV.append(TotV)
        galmass.append(log_Mstar_theory)

        print('Mhalo:', log_Mhalo_theory)
        print('Mstar', log_Mstar_theory)
        print('Rvir:', Rvir_calc(log_Mhalo_theory))
        print('c:', c_NFW(log_Mhalo_theory))
        print('Mdm:', Mdm_NFW_calc(radius, log_Mhalo_theory, c_NFW(log_Mhalo_theory)))
        print('Vdm:', V_dm_array)
        print('Vstar:', ML*Vstar_array)
        # Plotting rotation curve

        plt.figure(figsize = (15,15),dpi=120)
        plt.plot(radius, V_dm_array, 'k-', label = 'DM Velocity')
        plt.plot(radius, np.sqrt(ML)*Vstar_array, 'r-', label = 'Baryonic Velocity')
        plt.plot(radius, Total_Velocity, 'b-', label = 'Total velocity')
        plt.xlabel("Radius - kpc")
        plt.ylabel("Velocity - km/s")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper right", borderaxespad=0.)
        plt.title('Rotation curve, log(M$_{Star}$):'+str(log_Mstar_theory)+"[M$_{\odot}$] z="+str(redshift))
        plt.show()
    
############################### Plot a velocity function at certain radius


    ######################## Dutton Data
    xdata = np.arange(7,12)
    ydata = Dutton2010(xdata)
    
    
    ########################    


    
    totalV = [np.log10(x) for x in totalV] # log10 of velocity
    plt.figure(figsize = (12,12), dpi=60)
    plt.plot(galmass,totalV) # logVc vs log M*
    plt.plot(xdata,ydata) # Plot Dutton's relations
    plt.xlabel("log(M*/M$_{\odot}$)")
    plt.ylabel("log(V$_C$)")
    plt.show()
        
        
###################################################

if input('plot z=1?') == "Y":
    ############################# HYPERPARAMETERS
    redshift = 1.0

    "Choose model; NFW or DC14 or WDM"
    # Write the name of the model here. Ensure that it is written exactly as in the list
    model = "NFW"
    redshift = 1.0 #currently for 0.0 or 1.1

    wdm_dist = 0
    halo_dist = 0
    g_ML = 0
    g_Mhalo = 0
    Rd = 1.68
    al_gas = 1
    Ms = 5
    # Ms is the input free-streaming  scale for the WDM model

    h = 0.7

    G = 4.302*(10**(-6)) # The gravitational constant in Kpc/Msun(km/s)^2

    ML = 1.111308962394226

    #############################

    for i in range(len(Bins)):
        BinNumber = i
        log_Mhalo_theory = Bins[BinNumber][0] #Mhalo Bin
        log_Mstar_theory = Bins[BinNumber][1] #Mgal Bin
        # Range of radii to be plotted
        radius = np.arange(1, 30.1, 0.1)


        # Calculates values of Vstar as an array
        Vstar_array = np.array(Vstar_calc(radius, log_Mstar_theory)) # testwith swapping log Mstar with log Mgas
                                                                    #dark matter velocity ends up way too high


        # Turns Vdm into an array to be plotted
        V_dm_array = np.array(Vdm_calc(radius, log_Mhalo_theory, c_NFW(log_Mhalo_theory)))


        # Total velocity values to be plotted
        Total_Velocity = np.array(Vtot_lwrbnd(V_dm_array, Vstar_array, ML))

        print('Mhalo:', log_Mhalo_theory)
        print('Mstar', log_Mstar_theory)
        print('Rvir:', Rvir_calc(log_Mhalo_theory))
        print('c:', c_NFW(log_Mhalo_theory))
        print('Mdm:', Mdm_NFW_calc(radius, log_Mhalo_theory, c_NFW(log_Mhalo_theory)))
        print('Vdm:', V_dm_array)
        print('Vstar:', ML*Vstar_array)
        # Plotting rotation curve

        plt.figure(figsize = (15,15),dpi=120)
        plt.plot(radius, V_dm_array, 'k-', label = 'DM Velocity')
        plt.plot(radius, np.sqrt(ML)*Vstar_array, 'r-', label = 'Baryonic Velocity')
        plt.plot(radius, Total_Velocity, 'b-', label = 'Total velocity')
        plt.xlabel("Radius - kpc")
        plt.ylabel("Velocity - km/s")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper right", borderaxespad=0.)
        plt.title('Rotation curve, log(M$_{Star}$):'+str(log_Mstar_theory)+"[M$_{\odot}$] z="+str(redshift))
        plt.show()
###################################################
#Find Vc vs Radius for DC14 Model

if input('plot z=0,DC14:') == "Y":
    ############################# HYPERPARAMETERS
    redshift = 0.

    "Choose model; NFW or DC14 or WDM"
    # Write the name of the model here. Ensure that it is written exactly as in the list
    model = "DC14"
    redshift = 1.0 #currently for 0.0 or 1.1

    wdm_dist = 0
    halo_dist = 0
    g_ML = 0
    g_Mhalo = 0
    Rd = 1.68
    al_gas = 1
    Ms = 5
    # Ms is the input free-streaming  scale for the WDM model

    h = 0.7

    G = 4.302*(10**(-6)) # The gravitational constant in Kpc/Msun(km/s)^2

    ML = 1.111308962394226

    #############################

    for i in range(len(Bins)):
        BinNumber = i
        log_Mhalo_theory = Bins[BinNumber][0] #Mhalo Bin
        log_Mstar_theory = Bins[BinNumber][1] #Mgal Bin
        # Range of radii to be plotted
        radius = np.arange(1, 30.1, 0.1)


        # Calculates values of Vstar as an array
        Vstar_array = np.array(Vstar_calc(radius, log_Mstar_theory)) # testwith swapping log Mstar with log Mgas
                                                                    #dark matter velocity ends up way too high


        # Turns Vdm into an array to be plotted
        V_dm_array = np.array(Vdm_calc(radius, log_Mhalo_theory, c_DC14(c_NFW(log_Mhalo_theory),X(10**log_Mstar_theory,10**log_Mhalo_theory))))


        # Total velocity values to be plotted
        Total_Velocity = np.array(Vtot_lwrbnd(V_dm_array, Vstar_array, ML))
        # Plotting rotation curve

        plt.figure(figsize = (15,15),dpi=120)
        plt.plot(radius, V_dm_array, 'k-', label = 'DM Velocity')
        plt.plot(radius, np.sqrt(ML)*Vstar_array, 'r-', label = 'Baryonic Velocity')
        plt.plot(radius, Total_Velocity, 'b-', label = 'Total velocity')
        plt.xlabel("Radius - kpc")
        plt.ylabel("Velocity - km/s")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper right", borderaxespad=0.)
        plt.title('Rotation curve, log(M$_{Star}$):'+str(log_Mstar_theory)+"[M$_{\odot}$] z="+str(redshift))
        plt.show()
###################################################
#Find Vc at 2.2R for a range of Mgal

#Initial plot is Total velocity vs radius

#Total_Velocity = np.array(Vtot(V_dm_array, Vstar_array, ML))
redshift = 0.0

TotV = []
logMstr = []
for i in range(len(Bins)):
    BinNumber = i
    log_Mhalo_theory = Bins[BinNumber][0] #Mhalo Bin
    log_Mstar_theory = Bins[BinNumber][1] #Mgal Bin

    radius = [2.2]
    Vstar_array = np.array(Vstar_calc(radius, log_Mstar_theory))  
    V_dm_array = np.array(Vdm_calc(radius, log_Mhalo_theory, c_NFW(log_Mhalo_theory)))
    Total_Velocity = np.array(Vtot_lwrbnd(V_dm_array, Vstar_array, ML))
    TotV.append(np.log10(Total_Velocity))
    logMstr.append(log_Mstar_theory)
    print(Total_Velocity,Vstar_array,V_dm_array)

dutton2007x = np.linspace(8.2,11.8)
dutton2007y = np.linspace(1.6,2.48)

dutton2011x = np.linspace(8.1,11.9)
dutton2011y = np.linspace(1.6,2.6)

plt.figure(figsize = (15,15),dpi=120)
plt.plot(logMstr,TotV,label='FSmith')
plt.plot(dutton2007x,dutton2007y,label='Dutton 2007',linestyle='dashed')
plt.plot(dutton2011x,dutton2011y,label='Dutton 2011',linestyle='dashed')

plt.xlabel("log(M*)")
plt.ylabel("log$_{Vc}$(V$_c(2.2R))$Rotational Velocity - km/s")
plt.legend()
plt.show()

###################################################
radii = np.logspace(-2,2,50)

print(Bins[:][1])

for i in range(len(Bins)):
    if Bins[i][1] == 7.75:
        radius = radii
        BinNumber = i
        log_Mhalo_theory = Bins[BinNumber][0] #Mhalo Bin
        log_Mstar_theory = Bins[BinNumber][1] #Mgal Bin
        # Calculates values of Vstar as an array
        Vstar_array = np.array(Vstar_calc(radius, log_Mstar_theory)) # testwith swapping log Mstar with log Mgas
                                                                    #dark matter velocity ends up way too high
        # Turns Vdm into an array to be plotted
        V_dm_array = np.array(Vdm_calc(radius, log_Mhalo_theory, c_NFW(log_Mhalo_theory)))
        # Total velocity values to be plotted
        Total_Velocity = np.array(Vtot_lwrbnd(V_dm_array, Vstar_array, ML))

        radius = [np.log10(x) for x in radius]

        plt.figure(figsize = (15,15),dpi=120)
        plt.xlabel('log(R) [kpc]')
        plt.ylabel('log(v$_c$(R) [km s$^{-1}$]')
        plt.plot(radius, Total_Velocity, 'b-', label = 'Total velocity')
        plt.show()



###################################################
#Convert radius to R through Reff, plot figure 4 francesco unpublished"""