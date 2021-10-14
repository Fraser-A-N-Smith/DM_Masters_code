"""
Written by Max Dickson

Edited by FRASER SMTIH 2021
"""

import numpy as np
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from scipy.integrate import trapz
from scipy.special import erfc
import pandas as pd
import matplotlib.pyplot as plt
cosmo = cosmology.setCosmology("planck18")
Cosmo = cosmology.getCurrent()
plt.rcParams['font.size'] = 18 # changes font size
plt.rcParams['axes.linewidth'] = 2 # changes linewidth

def HMF(z , bins = 0.05 ,cum_var = 1.0):

    #Produces a Halo Mass Function (HMF) using a Tinker et al. (2008) HMF and a correction for subhalos from Behroozi et al. (2013)

    M = np.arange(8. , 15.5+bins ,bins) #Sets HMF mass range
    phi = mass_function.massFunction((10.0**M)*cosmo.h, z , mdef = '200m' , model = 'tinker08' , q_out="dndlnM") * np.log(10) * cosmo.h**3.0 * cum_var #Produces the Tinker et al. (2008) HMF

    a = 1./(1.+z) #Constants for Behroozi et al. (2013) Appendix G, eqs. G6, G7 and G8 subhalos correction
    C = np.power(10., -2.415 + 11.68*a - 28.88*a**2 + 29.33*a**3 - 10.56*a**4)
    logMcutoff = 10.94 + 8.34*a - 0.36*a**2 - 5.08*a**3 + 0.75*a**4
    correction = phi * C * (logMcutoff - M)

    return M , np.log10(phi + correction)

def derivative(x,y):

    #Returns the derivative of any input function

    func = interp1d(x,y,fill_value="extrapolate")
    dx = 0.1
    x_calc = np.arange(x[0],x[-1]+dx,dx)
    y_calc = func(x_calc)
    dydx = np.diff(y_calc)/np.diff(x_calc)

    dydx = np.append(dydx, dydx[-1]) #Preventing boundary abnormalities in the returned function
    dydx_low = np.mean(dydx[:10])
    dydx[0] = dydx_low
    dydx[1] = dydx_low
    dydx_high = np.mean(dydx[-10:])

    return interp1d(x_calc,dydx,fill_value=(dydx_low,dydx_high),bounds_error = False)

def integrals(SMF , HMF , sig):

    #Returns the integrals of the Stellar Mass Function (SMF) and HMF as calculated in Aversa et al. (2015) eq. 37

    M_s , phi_s = SMF[0] , SMF[1]
    M_h , phi_h = HMF[0] , HMF[1]
    phi_s , phi_h = 10.0**phi_s , 10.0**phi_h

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

def writematrix(SMF , sigma , z_range , title):

    #Writes a SMHM matrix to produce a SMHM for any redshift within z_range
    print("here")
    M_s , phi_s = SMF[0,:] , SMF[1,:]
    print("here2")
    abun = np.loadtxt("baldry_Bernardi_SMHM.txt") #Reads SMHM produced from abundance matching at z = 0.1
    print("here3")
    matrix = np.empty((0,M_s.size-1))
    print("here4")
    if type(sigma) == float:
        #Adaptation for the code to use constant and variable values of sigma
        sigma = np.repeat(sigma,z_range.size)
        print("here5")
    for i in range(z_range.size):
        #Calculates SMHM for each redshift in z_range
        print('start')
        z = z_range[i]
        sig_0 = sigma[i]
        M_h , phi_h = HMF(z)
        deri = derivative(abun[:,0],abun[:,1])(M_h)
        n=0
        e=1.
        print('hasnt crashed')
        while n < 3:
            #Calls the integrals function and matches integral values of SMF to interpolated values in the integral of HMF, iterating multiple times
            I_phi_s , I_phi_h , M_s_temp , M_h_temp = integrals(np.array([M_s , phi_s]) , np.array([M_h , phi_h]) , sig_0/deri)
            int_I_phi_h = interp1d(I_phi_h , M_h_temp , fill_value="extrapolate")
            M_h_match = np.array([])
            print('still nothing')
            for m in range(M_s_temp.size):
                M_h_match = np.append(M_h_match , int_I_phi_h(I_phi_s[m]))
            print('im surprised')
            M_s_iter , phi_s_iter = SMFfromSMHM(M_s_temp , M_h_match , sig_0 , z) #Reconstructing SMF
            int_phi_s_iter = interp1d(M_s_iter , phi_s_iter , fill_value="extrapolate")
            e_temp = max((phi_s - int_phi_s_iter(M_s))/phi_s) #Calculating relative error between reconstructed SMF and the input SMF
            print('if this prints i dont know where im going wrong')
            if e_temp < e:
                #Only accepts iterations that decrease the relative error
                e = e_temp
                deri = derivative(M_h_match,M_s_temp)(M_h)
            n += 1
            print(n)
        matrix = np.append(matrix,[M_h_match], axis=0) #Appending each halo mass array from the SMHM at each redshift to the matrix array
    matrix = np.append(matrix,[M_s_temp], axis=0) # Appending the stellar mass array to the end of the matrix array
    print('writing to file')
    with open(title+".txt" ,"w+") as file:
        #Writes the matrix to a file
        np.savetxt(file,np.array([z_range.size+1,M_s.size+1])[None],fmt="%i") #Writing the rows and columns of the file
        np.savetxt(file,np.append(z_range,np.inf)[None],fmt = "%f") #Writing the redshift of each SMHM
        np.savetxt(file,matrix.T) #Writing the SMHM matrix


def create_halo(minmass=8.0,maxmass=15.5,bin=0.1,scatter=0.0,redshift=0.1,vol=50):
    z = redshift
    h = Cosmo.h
    volume = vol**3 # volume vol on each side
    mass_range = np.arange(minmass, maxmass+bin, bin)
    hmf = mass_function.massFunction(10**mass_range*h, z, mdef='vir', model="tinker08", q_out='dndlnM') * np.log(10) * volume * bin * h**3
    return hmf






###############################################################################

hmftest = create_halo()
temp = len(hmftest) # assigned as temporary variable to increase efficieny of index matching
hmftest = [np.log10(x/temp) for x in hmftest]
plt.figure(dpi=120,figsize=(8,8))
plt.plot(np.arange(8.0,15.5+0.1,0.1),hmftest)
plt.show()
###############################################################################











HaloMassFunc = pd.read_csv("HaloMassFunction.txt",header=None,delim_whitespace=True)
HMF_M = [x for x in pd.Series(HaloMassFunc[0]).values]
HMF_P = [x for x in pd.Series(HaloMassFunc[1]).values]
HMF_standard = [HMF_M,HMF_P]


AM_HaoFu2020 = pd.read_csv('baldry_Bernardi_SMHM.txt',header=None,delim_whitespace=True)
AM_HaoFu2020_Halos = [x for x in pd.Series(AM_HaoFu2020[0]).values][2:] # remove the top two rows
AM_HaoFu2020_Halos = [float(x) for x in AM_HaoFu2020_Halos] # convert the strings into floats
AM_HaoFu2020_galaxies = [x for x in pd.Series(AM_HaoFu2020[1]).values][2:] # remove the top two rows
AM_HaoFu2020_galaxies = [float(x) for x in AM_HaoFu2020_galaxies] # convert the strings into floats






masses , phis = SMFfromSMHM(AM_HaoFu2020_galaxies, AM_HaoFu2020_Halos, 0.2, 0)
masses2 , phis2 = SMFfromSMHM(AM_HaoFu2020_galaxies, AM_HaoFu2020_Halos, 0.3, 0)
masses3 , phis3 = SMFfromSMHM(AM_HaoFu2020_galaxies, AM_HaoFu2020_Halos, 0.4, 0)
SMF1 = [masses, phis]

ShankarHalo = pd.read_csv('shankarhalos2021.csv',header=None,delim_whitespace=True)
ShankarHalo_M = [float(x) for x in pd.Series(ShankarHalo[0]).values]
ShankarHalo_P = [float(x) for x in pd.Series(ShankarHalo[1]).values]


Shalo = [[],[]]
Shalo[0] = ShankarHalo_M
Shalo[1] = ShankarHalo_P


plt.plot(ShankarHalo_M,ShankarHalo_P,label="shankar")
plt.plot(masses,phis,label="SMF")
plt.legend()
plt.show()

Hal1 = interp1d(ShankarHalo_P,ShankarHalo_M)
Stel1 = interp1d(phis,masses)
Stel2 = interp1d(phis2,masses2)
Stel3 = interp1d(phis3,masses3)

ranging = np.arange(min(phis),max(phis),0.25) # create interpolation range
ranging2 = np.arange(min(phis2),max(phis2),0.25)
ranging3 = np.arange(min(phis3),max(phis3),0.25)

Hal1_1 = Hal1(ranging)
Stel1_1 = Stel1(ranging)

Hal1_2 = Hal1(ranging2)
Stel1_2 = Stel2(ranging2)

Hal1_3 = Hal1(ranging3)
Stel1_3 = Stel3(ranging3)


Haloforintegration = [[],[]]
Haloforintegration[0] = Hal1_1
Haloforintegration[1] = ranging

Haloforintegration2 = [[],[]]
Haloforintegration2[0] = Hal1_2
Haloforintegration2[1] = ranging2

Haloforintegration3 = [[],[]]
Haloforintegration3[0] = Hal1_3
Haloforintegration3[1] = ranging3

SMFforintegration = [[],[]]
SMFforintegration[0] = Stel1_1
SMFforintegration[1] = ranging

SMFforintegration2 = [[],[]]
SMFforintegration2[0] = Stel1_2
SMFforintegration2[1] = ranging2

SMFforintegration3 = [[],[]]
SMFforintegration3[0] = Stel1_3
SMFforintegration3[1] = ranging3


plt.figure(dpi=100,figsize=(8,8))
plt.plot(Hal1_1,Stel1_1)
plt.show()


test = np.ndarray((2,len(Stel1_1)))
test[0] = Stel1_1
test[1] = ranging

"""
ShankarHalo = pd.read_csv('shankarhalos2021.csv',header=None,delim_whitespace=True)
ShankarHalo_M = [float(x) for x in pd.Series(ShankarHalo[0]).values]
ShankarHalo_P = [float(x) for x in pd.Series(ShankarHalo[1]).values]
"""

BaldryBernardi = pd.read_csv("baldry_bernardi_SMF_Max.txt",header=None,delim_whitespace=True)
#print(len(pd.Series(BaldryBernardi[0]).values))
BBtest= np.ndarray((2,129))
BBtest[0] = [float(x) for x in pd.Series(BaldryBernardi[0]).values]
BBtest[1] = [float(x) for x in pd.Series(BaldryBernardi[1]).values]
    
#BaldryBernardi02 = [Mg,Mh]
writematrix(BBtest, 0.2, np.array([0.0,0.5,1.0]), "testingpoint2")

a,b,c,d = integrals(SMFforintegration, Haloforintegration, 0) # changing sigma seems to have no effect
e,f,g,h = integrals(SMFforintegration2, Haloforintegration2, 0)
i,j,k,l = integrals(SMFforintegration3, Haloforintegration3, 0)

plt.figure(dpi=120,figsize=(12,12))
plt.plot(d,c,label="$\sigma$ = 0.2")
plt.plot(h,g,label="$\sigma$ = 0.3")
plt.plot(l,k,label="$\sigma$ = 0.4")
plt.legend()
plt.show()
#SMHM1 = integrals(SMF, HMF, sig)





scattered_hmf= pd.read_csv('scattered_halos.csv',header=None, delim_whitespace=(True))
hmasses_sctr = [x for x in pd.Series(scattered_hmf[1])]
phis_sctr = [x for x in pd.Series(scattered_hmf[0])]

scattered_hmf_ext= pd.read_csv('scattered_halos_ext.csv',header=None, delim_whitespace=(True))
hmasses_sctr_ext = [x for x in pd.Series(scattered_hmf_ext[1])]
phis_sctr_ext = [x for x in pd.Series(scattered_hmf_ext[0])]

phis_sctr = [np.log10((x/len(phis_sctr))) for x in phis_sctr]
phis_sctr_ext = [np.log10((x/len(phis_sctr_ext))) for x in phis_sctr_ext]

#plt.figure(dpi=120,figsize=(8,8))
#plt.scatter(hmasses_sctr,phis_sctr)
#plt.scatter(hmasses_sctr_ext,phis_sctr_ext)
#plt.show()


masses2 , phis2 = SMFfromSMHM(AM_HaoFu2020_galaxies, AM_HaoFu2020_Halos, 0.4, 0)
SMF2 = [masses2, phis2]
masses3 , phis3 = SMFfromSMHM(AM_HaoFu2020_galaxies, AM_HaoFu2020_Halos, 0.6, 0)
SMF3 = [masses3, phis3]

#print(max(masses),min(masses))


plt.figure(figsize = (12,12),dpi=60)
plt.xlabel('log(M*/M$_\odot)$')
plt.ylabel('log($\Phi$)')
plt.plot(np.arange(8.0,15.5+0.1,0.1),hmftest,label='GenCatalog')
plt.plot(masses,phis,label='SMF:0.2dex')
plt.plot(masses2,phis2,label='SMF:0.4dex')
plt.plot(masses3,phis3,label='SMF:0.6dex')
plt.plot(HMF_M,HMF_P,label='HMF')
plt.legend()
#plt.ylim(-5,-0.5)
plt.show()

SM1 = interp1d(phis,masses)
SM2 = interp1d(phis2, masses2)
SM3 = interp1d(phis3, masses3)

HMF_norm = interp1d(HMF_P,HMF_M)
HMF2 = interp1d(np.arange(8.0,15.5+0.1,0.1),hmftest)

range1 = np.arange(min(phis),max(phis),0.1) # setup a range of phis for the interpolation
range2 = np.arange(min(phis2),max(phis2),0.1) # setup a range of phis for the interpolation
range3 = np.arange(min(phis3),max(phis3),0.1) # setup a range of phis for the interpolation

x1 = [SM1(x) for x in range1] # interpolate for SM AM relations
x2 = [SM2(x) for x in range2] # interpolate for SM AM relations
x3 = [SM3(x) for x in range3] # interpolate for SM AM relations

y1 = [HMF_norm(y) for y in range1] # interpolate for HM AM relations
y2 = [HMF_norm(y) for y in range2] # interpolate for HM AM relations
y3 = [HMF_norm(y) for y in range3] # interpolate for HM AM relations

h1 = [Hal1(x) for x in range1]
h2 = [Hal1(x) for x in range2]
h3 = [Hal1(x) for x in range3]

hm1,hp1 = HMF(0)
hm2,hp2 = HMF(0,cum_var=1.2)
HAL2 = interp1d(hp1,hm1)
HAL2 = [HAL2(x) for x in range1]
HAL3  = interp1d(hp2,hm2)
HAL3_2 = [HAL3(x) for x in range2]
HAL3 = [HAL3(x) for x in range1]



plt.figure(dpi=120,figsize=(8,8))
plt.plot(h1,x1,label="$\sigma$: 0.2")
plt.plot(h2,x2,label="$\sigma$: 0.4")
plt.plot(h3,x3,label="$\sigma$: 0.6")
plt.plot(HAL2,x1,label="cumvar = 1,$\sigma$: 0.2")
plt.plot(HAL3,x1,label="cumvar = 1.2,$\sigma$: 0.2")
plt.plot(HAL3_2,x2,label="cumvar = 1.2, $\sigma$: 0.4")
plt.legend()
plt.xlabel("log(Mhalo/$M_\odot$)")
plt.ylabel("log(Mstar/$M_\odot$)")
plt.show()




#iphis , iphih, M_str, Mhal = integrals(SMF1, HMF_standard, 0.2)

#test = integrals(SMF1, HMF_standard, 0.2)[2]
#SMF = [[masses],[phis]]
#writematrix(SMF, 0.2, [0.0,0.2,0.4,0.6,0.8,1.0], "SMHMmatrix")

#if __name__ == "__main__":
