#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 17:45:52 2017

@author: pran
"""
"""
DATA ERROR ACCORDING TO DIFFERENT PSF ERROR LEVEL
"""

psftru = np.load('/Users/pran/Documents/code/new/data_psf_bad/PSF_tru.npy')
psfinterp = np.load('/Users/pran/Documents/code/new/data_psf_bad/PSF_interp.npy')
data_tru_1 = np.load('/Users/pran/Documents/code/new/data_psf_bad/galaxies_tru.npy')
data_tru_2 = np.load('/Users/pran/Documents/code/new/data_psf_bad/galaxies_tru.npy')
data_tru_3 = np.load('/Users/pran/Documents/code/new/data_psf_bad/galaxies_tru.npy')
data_tru_4 = np.load('/Users/pran/Documents/code/new/data_psf_bad/galaxies_tru.npy')
data_interp_1 = np.load('/Users/pran/Documents/code/new/data_output/psfinterp/1_psfinterp.npy')
data_interp_2 = np.load('/Users/pran/Documents/code/new/data_output/psfinterp/2_psfinterp.npy')
data_interp_3 = np.load('/Users/pran/Documents/code/new/data_output/psfinterp/3_psfinterp.npy')
data_interp_4 = np.load('/Users/pran/Documents/code/new/data_output/psfinterp/4_psfinterp.npy')

error1 = []
sum1 = 0
n1 = 0
error2 = []
sum2 = 0
n2 = 0
error3 = []
sum3 = 0
n3 = 0
error4 = []
sum4 = 0
n4 = 0

def errorfunction(image1, image2):
    
    res = np.linalg.norm(image2 - image1) ** 2 /np.linalg.norm(image1) ** 2
    
    return res
    
for i in range(515):
    
  res_psf = errorfunction(psftru[i], psfinterp[i])
   
  res_data = errorfunction(data_tru_4[i], data_interp_4[i])  #change 1 2 3 4 here 
  
  if 0 < res_psf < 0.1:
     
    n1 = n1 + 1
    sum1 = sum1 + res_data
    error1 = sum1/n1
    
  if 0.1 < res_psf < 0.3:
      
    n2 = n2 + 1
    sum2 = sum2 + res_data
    error2 = sum2/n2
    
  if 0.3 < res_psf < 0.5:
      
    n3 = n3 + 1
    sum3 = sum3 + res_data
    error3 = sum3/n3
    
  if 0.5 < res_psf:
      
    n4 = n4 + 1
    sum4 = sum4 + res_data
    error4 = sum4/n4
    
print 'finalnumber1', n1
print error1
print 'finalnumber2', n2
print error2
print 'finalnumber3', n3
print error3
print 'finalnumber4', n4
print error4

# for the interp 1 0.0357 0.1430 0.3008 0.4508
# for the interp 2 0.0308 0.1484 0.3021 0.4574
# for the interp 3 0.0306 0.1476 0.3015 0.4577
# for the interp 4 0.0345 0.1409 0.2917 0.4509

error0 = [0.146, 0.128, 0.126, 0.115]
error13 = [0.274, 0.259, 0.257, 0.250]
error35 = [0.430, 0.398, 0.397, 0.407]
error57 = [0.570, 0.6656, 0.523, 0.553]


plt.plot(snr, error0, 'p-')
plt.plot(snr, error13, 'g-')
plt.plot(snr, error35, 'r-')
plt.plot(snr, error57, 'y-')

plt.xlabel('SNR')
plt.ylabel('nmse')

plt.savefig('dataerror_diffenrencepsflevel.pdf')
plt.show()
