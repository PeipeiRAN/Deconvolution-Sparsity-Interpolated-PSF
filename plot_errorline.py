#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 12:05:02 2017

@author: pran
"""

"""
PLOT ERROR LINE
"""
import numpy as np
import math
import matplotlib.pyplot as plt

#nmse_psf_interp = [0.2451, 0.2233, 0.2214, 0.2172] #mean
#nmse_psf_true = [0.1315, 0.0957889, 0.0975306, 0.0957888]
#nmse_psf_interp = [0.1836, 0.1678, 0.1655, 0.1520] #median
#nmse_psf_true = [0.1148, 0.0752, 0.0726, 0.0725]

'''
===============
PLOT DATA ERROR
===============
'''
#error_data_PSF_true = [0.117, 0.077, 0.0742, 0.0726]
error_data_PSF_nochange = [0.138, 0.069, 0.066, 0.065]
error_data_PSF_true = [0.102, 0.063, 0.061, 0.060]
error_data_PSF_RCAcenter = [0.119, 0.066, 0.062, 0.062]
error_data_PSF_perfcenter = [0.113, 0.065, 0.061, 0.061]
error_data_PSF_Excenter = [ 0.119,0.069, 0.066, 0.065]
snr=[ 1, 2, 3, 5]

plt.figure(1)
plt.plot(snr, error_data_PSF_nochange,'-k', label ='using no updated PSF')
plt.plot(snr, error_data_PSF_true,'-r', label ='using true PSF')
plt.plot(snr, error_data_PSF_RCAcenter,'-g', label ='using RCAcenter PSF')
plt.plot(snr, error_data_PSF_perfcenter,'-b', label ='using perfcenter PSF')
plt.plot(snr, error_data_PSF_Excenter,'-c', label ='using Excenter PSF')
plt.xlabel('SNR')
plt.ylabel('error of data')
plt.legend(loc = 'upper right')
plt.savefig('compare_data_error_v1.pdf')
plt.show()

'''
==============
PLOT PSF ERROR
==============
'''
#error_psf_PSF_true = [0.00049, 8.347e-5, 5.176e-5, 4.432e-5]
error_psf_PSF_true = [0, 0, 0, 0]
error_psf_PSF_RCAcenter = [0.0260, 0.0273, 0.0272, 0.0273]
error_psf_PSF_perfcenter = [0.0012, 0.00032, 0.00024, 0.00022]
error_psf_PSF_Excenter = [0.035, 0.0345, 0.0353, 0.035]
snr=[ 1, 2, 3, 5]

plt.figure(2)
plt.plot(snr, error_psf_PSF_true,'-r', label ='using true PSF')
plt.plot(snr, error_psf_PSF_RCAcenter,'-g', label ='using RCAcenter PSF')
plt.plot(snr, error_psf_PSF_perfcenter,'-b', label ='using perfcenter PSF')
plt.plot(snr, error_psf_PSF_Excenter,'-c', label ='using Excenter PSF')
plt.xlabel('SNR')
plt.ylabel('error of psf')
plt.legend(loc = 'upper right')
plt.savefig('compare_psf_error_v1.pdf')
plt.show()
