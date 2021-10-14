import numpy as np
import matplotlib.pyplot as plt
from colossus.lss import mass_function
from scipy.interpolate import interp1d
import sys

from colossus.cosmology import cosmology
cosmology.setCosmology("planck18")
Cosmo = cosmology.getCurrent()
h = Cosmo.h

plt.ion()

plt.rc('font', family="serif")
plt.rc('text', usetex=True)



import warnings
warnings.filterwarnings("ignore") #ignore warnings



bin = 0.1
mass_range = np.arange(8., 15.5+bin, bin) # create bins

volume = 50**3 # volume 50 on each side
z = 0.1 # redshift

hmf_tinker = mass_function.massFunction(10**mass_range*h, z, mdef='vir', model="tinker08", q_out='dndlnM') * np.log(10) * volume * bin * h**3


#sat_corr = behroozi_shmf(mass_range, hmf_tinker, z)
hmf_tot = hmf_tinker #+ sat_corr

#print(hmf_tot,len(hmf_tot))

cumulative_mass_function = np.cumsum(hmf_tot) # cumulative sum to hmf_tot

print(cumulative_mass_function,len(cumulative_mass_function))
max_number = np.floor(np.max(cumulative_mass_function)) # find the maximum value in the list
#print(max_number)
if (np.random.uniform(0,1) > np.max(cumulative_mass_function)-max_number): # uniform distribution between 0 and 1
    max_number += 1 # create value 1 greater than the maximum value
    
interpolator = interp1d(cumulative_mass_function, mass_range, fill_value="extrapolate") # use extrapolation interpolation
range_numbers = np.random.uniform(np.min(cumulative_mass_function), np.max(cumulative_mass_function), int(max_number))

mhalo_catalog = interpolator(range_numbers)


#plt.figure(figsize=(8,8))
#plt.scatter(mass_range,cumulative_mass_function)
#plt.show()


import csv
with open('scattered_halos_ext.csv', 'w') as f:
   writer = csv.writer(f, delimiter='\t')
   writer.writerows(zip(range_numbers,mhalo_catalog))

#plt.figure(dpi=120,figsize=(8,8))
#plt.plot(mhalo_catalog,range_numbers)
#plt.show()
#plt.plot()
print(mhalo_catalog)
#print(mhalo_catalog,len(mhalo_catalog))
#print(mass_range,len(mass_range))
#print(cumulative_mass_function)
# array of central haloes masses

#plt.plot(cumulative_mass_function,mass_range)
#plt.show()
