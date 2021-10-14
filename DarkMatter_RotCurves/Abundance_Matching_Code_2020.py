#Abundance Matching Code 2020

############################################################
#IMPORTS
############################################################

import matplotlib # imper nescessary imports
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit
import scipy.integrate as integrate
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import scipy.interpolate as interp


#############################################################
#DEFINE FUNCTIONS
#############################################################

def abundancematch(x,y,z): # modular function that works out the 
#x value of requested y values, thus being effective for abundance matching code
	#x = [abs(i) for i in x] # this has minimal to no effect on results, it can be
	#y = [abs(j) for j in y] # commented out
	#z = [abs(k) for k in z]
	#print(y)
	"""
	y = y[:-15]
	x = x[:-15]

	for i in y:
		if i > max(z)-0.01 or i<min(z)+0.01:
			x = [j for j in x if j!=i]
			y = [k for k in y if k!=i]
"""
	y,x = zip(*sorted(zip(y,x)))
	#print(y)
	f = interp1d( y, x )#,fill_value='extrapolate')#,kind='cubic')
	#experimenting with the kind='' argument in scipys interpolation procedure
	#https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html

	#attempt zipping and sorting the x and y arrays
	result = f(z)
	#print(len(z),len(result))
	return result

def PGrylls2019(M_halo,z,N_z,M_n_z,Beta_z,Gamma_z,N_01,M_n_01,Beta_01,Gamma_01):

	#Abundance matching functional form taken from Pips Paper: 
	#Predicting fully self-consistent satellite richness, galaxygrowth and starformation 
	#rates from the STasticalsEmi-Empirical modeL steel.Philip J. Grylls

	N = N_01 + N_z*((z-0.1)/(z+1)) # N as a function of z

	M_n = M_n_01 + M_n_z*((z-0.1)/(z+1)) # M_n as a function of z

	Beta = Beta_01 + Beta_z*((z-0.1)/(z+1)) # Beta as a function of z

	Gamma = Gamma_01 + Gamma_z*((z-0.1)/(z+1)) # Gamma as a function of z

	M_star = 2.0 * M_halo * N * ((M_halo/10.**M_n)**(-1.0*Beta) + (M_halo/10.**M_n)**Gamma)**(-1.0)

	return M_star # equations taken from pip's paper with the denominatiors slightly altered


def plotmodels(Mhalo,Mgalaxy,Models,error): # plots defined models for Mhalo vs Mgalaxy
	#requires nested list of lists of the Mhaloes and Mgalaxies in order of list of models
	for i in range(Models):

		plt.figure(figsize = (20,20),dpi=60) #sets figure dimensions
		plt.plot(Mhalo[i],Mgalaxy[i],label=str(Model))

	return

def importMhaloMstarBen(model): #reads in the Mhalo Mstar relationships depending on state model type

	data = open(str(model)+'_MstarMhalo.dat','r') # open .dat file starting with model name
	if data.mode == 'r':
		contents = data.read()
		contents = contents.split(']')
		model_starmasses = list(map(float,contents[0][1:].split()))
		model_halomasses = list(map(float,contents[1][2:].split())) 

	return model_starmasses, model_halomasses 

def NicolasDataset(datasetname):
	data1 = pd.read_csv(datasetname[0])
	data2 = pd.read_csv(datasetname[1])
	return data1, data2

def PGrylls2019(M_halo,z,N_z,M_n_z,Beta_z,Gamma_z,N_01,M_n_01,Beta_01,Gamma_01):

	#Abundance matching functional form taken from: 
	#Predicting fully self-consistent satellite richness, galaxygrowth and starformation 
	#rates from the STasticalsEmi-Empirical modeL steel.Philip J. Grylls

	N = N_01 + N_z*((z-0.1)/(z+1)) # N as a function of z

	M_n = M_n_01 + M_n_z*((z-0.1)/(z+1)) # M_n as a function of z

	Beta = Beta_01 + Beta_z*((z-0.1)/(z+1)) # Beta as a function of z

	Gamma = Gamma_01 + Gamma_z*((z-0.1)/(z+1)) # Gamma as a function of z

	M_star = 2.0 * M_halo * N * ((M_halo/10.**M_n)**(-1.0*Beta) + (M_halo/10.**M_n)**Gamma)**(-1.0)

	return M_star

def AbundanceMatch(x,y,arr_range):
	#interpolate arrays out to equal length values between galaxy limits
	#then compare to each other and obtain ms values at each phi, to get
	#abundance matching relation.
	f1 = interp1d(x,y, kind='linear')
	xnew = np.linspace(x[0],y[-1],100)
	plt.plot(x,y,label='original')
	plt.plot(xnew,f1(xnew),label='new')
	plt,legend()
	plt.show()
	print(arr_range)
	print()
	print(f1(xnew))
	
	return 



def remove_excess(Halo_logMs,Halo_log_Cuml,Gal_logMs):
	#print(len(Halo_logMs))
	for i in range(len(Halo_logMs)):
		if Halo_logMs[i] > (Gal_logMs[0]-0.1):
			Halo_log_Cuml = Halo_log_Cuml[i:]
			Halo_logMs = Halo_logMs[i:]
			#print(len(Halo_logMs))
			break
	return Halo_logMs , Halo_log_Cuml

def interpolatefory(x,y,galaxy): # investigate using this as a solution 
											#to interpolation issue 7/9/20
	x,y = remove_excess(x,y,galaxy)

	interp_halo = interp.interp1d(y,x) # create an interpolation of the halo
	Mass_halo = interp_halo(galaxy) #withdraw the x values of the interpolation

	return Mass_halo


################################################################
#DEFINE DISPLAY PARAMETERS
################################################################

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18} # font setup for matplotlib


matplotlib.rc('font', **font) # change matplotlib font
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small') # change the sizes and fonts of displays in the script
plt.rc('ytick', labelsize='x-small')


###############################################################
#READ IN DATASETS
###############################################################

#this section reads in the model names which are used to find the correct file names in the directory
#to take as inputs

models = ['DC14','NFW','WDM1','WDM2','WDM3'] # models CDM and WDM # DeCinto2014 and navaro franko weltz
models_Nicola = ['CDM','3kev','SN1','SN2'] # Models for data provided by nicola


galaxy_data = pd.read_csv('Bernardi13MFsSerExp.dat',header=None,delim_whitespace=True) #bernardi 2013 data for galaxy data
Gal_lowmass_Baldry = pd.read_csv('Baldry2012.dat',header=None,skiprows=1,delim_whitespace=True) # data from baldry and papstergis
Gal_lowmass_Papastergis = pd.read_csv('Papastergis2013.dat',header=None,skiprows=1,delim_whitespace=True)# 2012 and 2013 for the low mass
																				# dwarf glaxy range to examine Mhalo Mgal relationship 
CDM = pd.read_csv('HaloMassFunction.txt',header=None,delim_whitespace=True) #CDM data provided by Francesco Shankak
CDM_Nicola = pd.read_csv('tab_CDM.dat',header=None,delim_whitespace=True) #Nicola's Version of CDM
KeV3_Nicola = pd.read_csv('tab_3kev.dat',header=None,delim_whitespace=True)#3KeV data Nicola
ResProd_Nicola = pd.read_csv('tab_RP_7keV_2e-10.dat',header=None,delim_whitespace=True) # Resonant Production Nicola
MixAngle_Nicola = pd.read_csv('tab_RP_7keV_5e-11.dat',header=None,delim_whitespace=True) # Mixing Angle Nicola
#why is it called mixing angle
###################################################################
#Set up calculation for galaxies
##################################################################
#BERNARDI BALDRY AND PAPASTERGIS DATA
Gal_logMs = pd.Series(galaxy_data[0]).values #1st column is logMs, galaxy data refers to the bernardi dataset
Gal_logPhiTot = pd.Series(galaxy_data[1]).values #2nd column is log(Phi_total)

Gal_Baldry_logMs = pd.Series(Gal_lowmass_Baldry[0]).values # aquire values from the baldry dataset
Gal_Baldry_Phi = pd.Series(Gal_lowmass_Baldry[2]).values*(10**-3.0) # look at why values have been multiplied by 10^-3
Gal_Baldry_logPhi = np.log10(Gal_Baldry_Phi)


Gal_Papastergis_logMs = pd.Series(Gal_lowmass_Papastergis[0]).values # aquire values for the papastergis data
Gal_Papastergis_Phi = pd.Series(Gal_lowmass_Papastergis[2]).values
Gal_Papastergis_logPhi = np.log10(Gal_Papastergis_Phi)

###########################################################
#Assemble hybrid arrays of papastergis, baldry, and bernardi data.
#in order to extend galaxy data to that of lower masses.

Gal_PhiTot = []
for i in Gal_logPhiTot: # cycle through the recorded logged values of phi and unlog them
	Gal_PhiTot.append(10**i)# WHY IS THIS THE CASE?
							#its to unlog the data and put it back into SI units not dex

Gal_Hybrid_PhiTot = np.append(Gal_Baldry_Phi[3:12],Gal_PhiTot) # form hybrid array of baldry and bernardi
Gal_Hybrid_logMs = np.append(Gal_Baldry_logMs[3:12],Gal_logMs) # datasets 

Gal_Hybrid_log_PhiTot= [np.log10(x) for x in Gal_Hybrid_PhiTot]

Gal_Hybrid_PhiTot_alternat = np.append(Gal_Papastergis_Phi[0:3],Gal_PhiTot) # form hybrid array of bernardi and papastergis
Gal_Hybrid_logMs_alternat = np.append(Gal_Papastergis_logMs[0:3],Gal_logMs) # to be alternate 

Gal_Hybrid_log_PhiTot_alternat = [np.log(x) for x in Gal_Hybrid_PhiTot_alternat]

Gal_Hybrid_PhiTot_alternat2 = np.append(Gal_Baldry_Phi[0:12],Gal_PhiTot) # form hybrid array of baldry and bernardi
Gal_Hybrid_logMs_alternat2 = np.append(Gal_Baldry_logMs[0:12],Gal_logMs) # datasets 

Gal_Hybrid_log_PhiTot_alternat2 = [np.log10(x) for x in Gal_Hybrid_PhiTot_alternat2]

Gal_Cuml = []
Gal_log_Cuml = []

Gal_Cuml_alternat = []
Gal_Cuml_alternat2 = []
Gal_log_Cuml_alternat = [] # prep cumulative arrays to put calculated cuml data in 
Gal_log_Cuml_alternat2 = []

for i in range(len(Gal_Hybrid_logMs_alternat)):
	Gal_Cuml_alternat.append(np.trapz(Gal_Hybrid_PhiTot_alternat[i:],Gal_Hybrid_logMs_alternat[i:]))
	Gal_log_Cuml_alternat.append(np.log10(Gal_Cuml_alternat[-1])) # form cumulative arrays for probability density
														#and perform the trapezium rule on the alternates
for i in range(len(Gal_Hybrid_logMs_alternat2)):
	Gal_Cuml_alternat2.append(np.trapz(Gal_Hybrid_PhiTot_alternat2[i:],Gal_Hybrid_logMs_alternat2[i:]))
	Gal_log_Cuml_alternat2.append(np.log10(Gal_Cuml_alternat2[-1]))

for i in range(len(Gal_Hybrid_logMs)):
	Gal_Cuml.append(np.trapz(Gal_Hybrid_PhiTot[i:],Gal_Hybrid_logMs[i:]))
	Gal_log_Cuml.append(np.log10(Gal_Cuml[-1]))

############################################
#plot hybrid galaxy and alternate hybrid galaxy arrays, against
#their respective halo masses
############################################


plt.figure(figsize = (12,12),dpi=120)
plt.plot(Gal_Hybrid_logMs,Gal_log_Cuml)
plt.plot(Gal_Hybrid_logMs_alternat,Gal_log_Cuml_alternat)
plt.plot(Gal_Hybrid_logMs_alternat2,Gal_log_Cuml_alternat2)
plt.scatter(Gal_Hybrid_logMs,Gal_log_Cuml,marker='x',label='Bernardi2013+Baldry2012')
plt.scatter(Gal_Hybrid_logMs_alternat,Gal_log_Cuml_alternat,marker='o',label='Bernardi2013+Papastergis2013')
plt.scatter(Gal_Hybrid_logMs_alternat2,Gal_log_Cuml_alternat2,marker='o',label='Bernardi2013+Baldry2012')
plt.legend()
plt.ylabel('$log(\Phi(M>)_{Gal})[Mpc^{-3}dex^{-1}]$')
plt.xlabel('$log(M_{Gal}/M_{\odot}) $')
plt.show()


#############################################
#Plot Galaxy Relationship
#############################################

#plot the bernardi, papastergis and baldry data to achieve lower mass dwarf galaxies

plt.figure(figsize = (12,12),dpi=120)
plt.scatter(Gal_logMs,Gal_logPhiTot,marker='x',label='Bernardi2013')
plt.scatter(Gal_Papastergis_logMs,Gal_Papastergis_logPhi,marker='o',label='Papastergis2013')
plt.scatter(Gal_Baldry_logMs,Gal_Baldry_logPhi,marker='^',label='Baldry2012')
plt.legend()
plt.ylabel('$log(\Phi(M)_{Gal})[Mpc^{-3}dex^{-1}]$')
plt.xlabel('$log(M_{Gal}/M_{\odot}) $')
plt.show()


#####################################################
#Calculation of different models
#####################################################
#we get the probability or number density on the y axis
#then putting the halo mass on the x axis. We can plot this on a cumulative graph 
#that will show the different dark matter models with their respective mass
#to probabiliry relationships. We then evaluate the galaxyy and halo mass functions
#together to find the mass relationship between halo and galaxies residing within


"""
Nicola's Email
I attach below 4 data files with the differential and cumulative mass function for 
CDM, WDM with particle mass 3 keV, and for two sterile neutrino models (resonant 
production with particle mass 7 keV and mixing angle 2e-10 and 5e-11). 
Each file contains: 
Column 1 - Mhalo (in units 10ˆ12 Msun)
Column 2 - Cumulative Halo mass function 
Column 3 - Differential Halo mass function 
Column 4 - Concentration parameter correspnding to Halo
In the other columns there are quantities not relevant for the moment. 
All the best
Nicola
"""

#####################################################
#Set up calculation for CDM
#####################################################

#CDM SHANKAR 2019####################################

CDM_cumulative_shankar = [] 
CDM_log10cumulative_shankr =[]

#CDM_logMs = pd.Series(CDM[0]).values #draw values from 2 colums in halomassfunction.txt
#CDM_logPhiTot = pd.Series(CDM[1]).values

CDM_logMs = pd.Series(CDM[0]).values
CDM_logPhiTot = pd.Series(CDM[1]).values
CDM_PhiTot = []

for i in CDM_logPhiTot:
	CDM_PhiTot.append((10**i))

for i in range(len(CDM_logMs)): # use trapezium rule to generate cumulative data
	CDM_cumulative_shankar.append(np.trapz(CDM_PhiTot[i:],CDM_logMs[i:])) 
	CDM_log10cumulative_shankr.append(np.log10(CDM_cumulative_shankar[-1]))

#CDM Nicola 2019#####################################



CDM_Ms_Nicola = pd.Series(CDM_Nicola[0]).values # read values out of .dat files in directory in solar units/10**12
CDM_Cumulative_Nicola = pd.Series(CDM_Nicola[1]).values #take values Nicola says are Cumulative

CDM_log10_Ms_Nicola = [np.log10(i*10.0**12.0) for i in CDM_Ms_Nicola] # Conversion to SI units
CDM_log10_Cumulative_Nicola = [np.log10(i) for i in CDM_Cumulative_Nicola]#log all values base 10
#for i in range(len(CDM_logMs_Nicola))
#i = i*10.0**12.0 #conversion to SI units. Check Nicolas email for details


#######################################################
#SET UP CALCULATION FOR 3KeV
#######################################################

KeV3_Ms_Nicola = pd.Series(KeV3_Nicola[0]).values # read values out of .dat files in directory in solar units/10**12
KeV3_Cumulative_Nicola = pd.Series(KeV3_Nicola[1]).values #take values Nicola says are Cumulative

KeV3_log10_Ms_Nicola = [np.log10(i*10.0**12.0) for i in KeV3_Ms_Nicola] # Conversion to SI units
KeV3_log10_Cumulative_Nicola = [np.log10(i) for i in KeV3_Cumulative_Nicola]#log all values base 10

########################################################
#SET UP CALCULATION FOR RESONANT PRODUCTION
#####################################################

#7KeV Resonant Production 2e-10
ResProd_Ms_Nicola = pd.Series(ResProd_Nicola[0]).values # read values out of .dat files in directory in solar units/10**12
ResProd_Cumulative_Nicola = pd.Series(ResProd_Nicola[1]).values #take values Nicola says are Cumulative

ResProd_log10_Ms_Nicola = [np.log10(i*10.0**12.0) for i in ResProd_Ms_Nicola] # Conversion to SI units
ResProd_log10_Cumulative_Nicola = [np.log10(i) for i in ResProd_Cumulative_Nicola]#log all values base 10

#7KeV Resonant Production 5e-11
MixAngle_Ms_Nicola = pd.Series(MixAngle_Nicola[0]).values # read values out of .dat files in directory in solar units/10**12
MixAngle_Cumulative_Nicola = pd.Series(MixAngle_Nicola[1]).values #take values Nicola says are Cumulative

MixAngle_log10_Ms_Nicola = [np.log10(i*10.0**12.0) for i in MixAngle_Ms_Nicola] # Conversion to SI units
MixAngle_log10_Cumulative_Nicola = [np.log10(i) for i in MixAngle_Cumulative_Nicola]#log all values base 10


#####################################################
#Plot cumulative graph
#####################################################
plt.figure(figsize = (12,12),dpi=100)
plt.plot(CDM_logMs,CDM_log10cumulative_shankr,label='Shankar CDM')
plt.plot(CDM_log10_Ms_Nicola,CDM_log10_Cumulative_Nicola,label='Nicola CDM') 
plt.plot(KeV3_log10_Ms_Nicola,KeV3_log10_Cumulative_Nicola,label='Nicola 3KeV')
plt.plot(ResProd_log10_Ms_Nicola,ResProd_log10_Cumulative_Nicola,label='ResProd 7KeV 2e-10')
plt.plot(MixAngle_log10_Ms_Nicola,MixAngle_log10_Cumulative_Nicola,label='ResProd 7KeV 5e-11')
plt.plot(Gal_Hybrid_logMs,Gal_log_Cuml,label='galaxy 1st config') # adding galaxy data to cumulative plot
plt.plot(Gal_Hybrid_logMs_alternat,Gal_log_Cuml_alternat,label='galaxy 2nd config')
# tbis plot shows CDM being different for
#shankar and nicola. Ask francesco the reason this is different

#plot currently shows models not diverging fast enough. need to see validity of Nicolas models
plt.legend()
plt.show()

#####################################################
#Conduct Abundance Matching for initial hybrid galaxy data array


CDM_AbMtch = abundancematch(CDM_logMs,CDM_log10cumulative_shankr,Gal_log_Cuml)
CDM_AbMtch_Nicola = abundancematch(CDM_log10_Ms_Nicola,CDM_log10_Cumulative_Nicola,Gal_Hybrid_log_PhiTot)
KeV3_AbMtch_Nicola = abundancematch(KeV3_log10_Ms_Nicola,KeV3_log10_Cumulative_Nicola,Gal_Hybrid_log_PhiTot)
ResonantProduction_Nicola = abundancematch(ResProd_log10_Ms_Nicola,ResProd_log10_Cumulative_Nicola,Gal_Hybrid_log_PhiTot)
MixingAngle = abundancematch(MixAngle_log10_Ms_Nicola,MixAngle_log10_Cumulative_Nicola,Gal_Hybrid_log_PhiTot)
#when the hybrid array was used caused out of bounds error to be raised 
#investigate error due to hybrid array

plt.plot(CDM_AbMtch,Gal_Hybrid_logMs,label='shankar cdm')
plt.plot(CDM_AbMtch_Nicola,Gal_Hybrid_logMs, label='nicola cdm')
plt.plot(KeV3_AbMtch_Nicola,Gal_Hybrid_logMs,label='nicola 3KeV')
plt.plot(ResonantProduction_Nicola,Gal_Hybrid_logMs,label='7KeV RP 2x10$^{-10}$')
plt.plot(MixingAngle,Gal_Hybrid_logMs,label='7KeV RP 2x10$^{-10}$')
plt.ylabel('$log_{10}[M_{*}/M_{\odot}]$')
plt.xlabel('$log_{10}[M_{Halo}/M_{\odot}]$')
plt.legend()
plt.show()

#####################################################
#Conduct Abundance Matching for second hybrid galaxy data array
#print(CDM_log10_Cumulative_Nicola,CDM_log10cumulative_shankr)

CDM_AbMtch2= abundancematch(CDM_logMs,CDM_log10cumulative_shankr,Gal_log_Cuml_alternat2)
CDM_AbMtch_Nicola2 = abundancematch(CDM_log10_Ms_Nicola,CDM_log10_Cumulative_Nicola,Gal_Hybrid_log_PhiTot_alternat2)
KeV3_AbMtch_Nicola2 = abundancematch(KeV3_log10_Ms_Nicola,KeV3_log10_Cumulative_Nicola,Gal_Hybrid_log_PhiTot_alternat2)
ResonantProduction_Nicola2 = abundancematch(ResProd_log10_Ms_Nicola,ResProd_log10_Cumulative_Nicola,Gal_Hybrid_log_PhiTot_alternat2)
MixingAngle2 = abundancematch(MixAngle_log10_Ms_Nicola,MixAngle_log10_Cumulative_Nicola,Gal_Hybrid_log_PhiTot_alternat2)
#when the hybrid array was used caused out of bounds error to be raised 
#investigate error due to hybrid array

plt.plot(CDM_AbMtch2,Gal_Hybrid_logMs_alternat2,label='shankar cdm')
#plt.plot(CDM_AbMtch_Nicola2,Gal_Hybrid_logMs_alternat2, label='nicola cdm')
plt.plot(KeV3_AbMtch_Nicola2,Gal_Hybrid_logMs_alternat2,label='nicola 3KeV')
plt.plot(ResonantProduction_Nicola2,Gal_Hybrid_logMs_alternat2,label='7KeV RP 2x10$^{-10}$')
plt.plot(MixingAngle2,Gal_Hybrid_logMs_alternat2,label='7KeV RP 2x10$^{-10}$')
plt.ylabel('$log_{10}[M_{*}/M_{\odot}]$')
plt.xlabel('$log_{10}[M_{Halo}/M_{\odot}]$')
plt.legend()
plt.show()

####################################################
#Plot with error areas

error = 0.2 #dex

plt.plot(CDM_AbMtch2,Gal_Hybrid_logMs_alternat2,label='shankar cdm')
plt.fill_between(CDM_AbMtch2,Gal_Hybrid_logMs_alternat2+error,Gal_Hybrid_logMs_alternat2-error,alpha=0.1)
#plt.plot(CDM_AbMtch_Nicola2,Gal_Hybrid_logMs_alternat2, label='nicola cdm')
#plt.fill_between(CDM_AbMtch_Nicola2,Gal_Hybrid_logMs_alternat2+error,Gal_Hybrid_logMs_alternat2-error,alpha=0.1)
plt.plot(KeV3_AbMtch_Nicola2,Gal_Hybrid_logMs_alternat2,label='nicola 3KeV')
plt.fill_between(KeV3_AbMtch_Nicola2,Gal_Hybrid_logMs_alternat2+error,Gal_Hybrid_logMs_alternat2-error,alpha=0.1)
plt.plot(ResonantProduction_Nicola2,Gal_Hybrid_logMs_alternat2,label='7KeV RP 2x10$^{-10}$')
plt.fill_between(ResonantProduction_Nicola2,Gal_Hybrid_logMs_alternat2+error,Gal_Hybrid_logMs_alternat2-error,alpha=0.1)
plt.plot(MixingAngle2,Gal_Hybrid_logMs_alternat2,label='7KeV RP 2x10$^{-10}$')
plt.fill_between(MixingAngle2,Gal_Hybrid_logMs_alternat2+error,Gal_Hybrid_logMs_alternat2-error,alpha=0.1)
plt.ylabel('$log_{10}[M_{*}/M_{\odot}]$')
plt.xlabel('$log_{10}[M_{Halo}/M_{\odot}]$')
plt.legend()
plt.show()