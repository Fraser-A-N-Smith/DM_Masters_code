import numpy as np
import pandas as pd
from colossus.cosmology import cosmology
import matplotlib.pyplot as plt
cosmology.setCosmology('planck15')
from colossus.lss import mass_function
from scipy.interpolate import interp1d

redshift = 0.0

plt.rcParams['font.size'] = 18 # changes font size
plt.rcParams['axes.linewidth'] = 2 # changes linewidth

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
def PGrylls19(mhalo,z,Mn01,Mnz,N01,Nz,B01,Bz,G01,Gz):
    
    Mstar = 2.0*Mhalo*N(z,N01,Nz)*((Mhalo/Mn(z,Mn01,Mnz))**(-Beta(z,B01,Bz))+(Mhalo/Mn(z,Mn01,Mnz))**(Gamma(z,G01,Gz)))**(-1.0)
    
    return Mstar

#generate the halo mass function of central haloes (no subhaloes)
mina=10.
maxa=15.1
binsize=0.1
z=0.1
Vol=500.**3.
M = 10**np.arange(mina,maxa, binsize)
mfunc = mass_function.massFunction(M, z, mdef = '200m', model = 'tinker08', q_out = 'dndlnM')
plt.plot(np.log10(M), np.log10(mfunc))

#compute the cumulative halo mass function and extract the haloes
a=mfunc[::-1]
a=np.cumsum(a*binsize)
a=a[::-1]
vecint=np.arange(np.int(np.max(a*Vol)))
b=a[::-1]
cumh=np.log10(M[::-1])
Mhalo=np.interp(vecint,b*Vol,cumh)
#plt.plot(np.log10(M), np.log10(a*Vol))

#bin the extracted halo catalogue as dN/dM/Vol to get a halo mass function which must match the input one! 
#use fast histogram method
hist = np.histogram(Mhalo, bins=len(M),range=(mina,maxa))[0] 
plt.scatter(np.log10(M),np.log10(hist/binsize/Vol))
#use brute-force method
Mlog=np.log10(M)
maccm=Mlog
maccp=maccm+binsize
maccs=0.5*(maccm+maccp)
#Mhalomean=0.*macc
Num=0.*M
for i in range(len(M)):
    ix=Mhalo[(Mhalo>=maccm[i]) & (Mhalo<maccp[i])]
    Num[i]=np.float(len(ix))
plt.plot(maccs,np.log10(Num/binsize/Vol),'r+')
plt.xlabel('log(M$_{Halo}$/M$_\odot$)')
plt.show()

#assign galaxies to haloes
#Mgal=Grylls19(z,Mhalo,0.15)

AM_HaoFu2020 = pd.read_csv('baldry_Bernardi_SMHM.txt',header=None,delim_whitespace=True)
AM_HaoFu2020_Halos = [x for x in pd.Series(AM_HaoFu2020[0]).values][2:] # remove the top two rows
AM_HaoFu2020_Halos = [float(x) for x in AM_HaoFu2020_Halos] # convert the strings into floats
AM_HaoFu2020_galaxies = [x for x in pd.Series(AM_HaoFu2020[1]).values][2:] # remove the top two rows
AM_HaoFu2020_galaxies = [float(x) for x in AM_HaoFu2020_galaxies] # convert the strings into floats


Mgal= Moster(Mhalo,0)
step=0.3
mina=8.0
maxa=12.
maccm=np.arange(mina,maxa,step)
maccp=maccm+step
macc=0.5*(maccm+maccp)
Mhalomean=0.*macc

#compute inverse relation: Mhalo at fixed M*
for i in range(len(macc)):
    Mfin=np.mean(Mhalo[(Mgal>=maccm[i]) & (Mgal<maccp[i])])
    Mhalomean[i]=Mfin
print('working')
plt.figure(dpi=120,figsize=(12,12))
plt.plot(macc,Mhalomean,label='Moster')

MgalHao = interp1d(AM_HaoFu2020_Halos,AM_HaoFu2020_galaxies)
Mgal= MgalHao(Mhalo)
step=0.3
mina=8.
maxa=12.
maccm=np.arange(mina,maxa,step)
maccp=maccm+step
macc=0.5*(maccm+maccp)
Mhalomean=0.*macc
print(macc,Mhalomean)
#compute inverse relation: Mhalo at fixed M*
for i in range(len(macc)):
    Mfin=np.mean(Mhalo[(Mgal>=maccm[i]) & (Mgal<maccp[i])])
    Mhalomean[i]=Mfin
print('working')
plt.plot(macc,Mhalomean,label='Baldry-Bernardi')

plt.legend()
plt.ylabel('log((M$_{Halo}$/M$_\odot$)')
plt.xlabel('log(Mgal/$M_\odot$)')


point2leja = pd.read_csv('(FR) Leja SMHM sigma = 0.2.txt',header=None,delim_whitespace=True,skiprows=2)
point2lejaMh = [x for x in pd.Series(point2leja[0]).values]
point2lejaGl = [x for x in pd.Series(point2leja[1]).values]

Mgalleja = interp1d(point2lejaMh,point2lejaGl)
Mgal= Mgalleja(Mhalo)
step=0.3
mina=8.
maxa=12.
maccm=np.arange(mina,maxa,step)
maccp=maccm+step
macc=0.5*(maccm+maccp)
Mhalomean=0.*macc
print(macc,Mhalomean)
#compute inverse relation: Mhalo at fixed M*
for i in range(len(macc)):
    Mfin=np.mean(Mhalo[(Mgal>=maccm[i]) & (Mgal<maccp[i])])
    Mhalomean[i]=Mfin
plt.plot(macc,Mhalomean,label='leja')

"""
Mgal= PGrylls19(Mhalo,0.1, 11.91,0.52,0.029,-0.018,2.09,-1.03,0.64,0.084)
step=0.3
mina=9.
maxa=12.
maccm=np.arange(mina,maxa,step)
maccp=maccm+step
macc=0.5*(maccm+maccp)
Mhalomean=0.*macc
print(macc,Mhalomean)
#compute inverse relation: Mhalo at fixed M*
for i in range(len(macc)):
    Mfin=np.mean(Mhalo[(Mgal>=maccm[i]) & (Mgal<maccp[i])])
    Mhalomean[i]=Mfin
print('working')
plt.plot(macc,Mhalomean)
"""



plt.legend()
plt.show()

