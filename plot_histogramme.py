#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 11:24:52 2017

@author: pran
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import random

""" INPUT IMAGE """

psfinterp = np.load('/Users/pran/Documents/code/new/data_psf_bad/PSF_interp.npy')
psffinal = np.load('/Users/pran/Documents/code/new/psffinal.npy')
psftru = np.load('/Users/pran/Documents/code/new/data_psf_bad/PSF_tru.npy')

""" PLOT HISTOGRAM """

res0 = (np.array([np.linalg.norm(x) ** 2 for x in (psfinterp - psftru)]) /
               np.array([np.linalg.norm(x) ** 2 for x in psftru]))
plt.hist(res0,bins = 'auto', label = 'initial')
plt.xlabel('PSF errorhistogram')
plt.savefig('initial PSF errorhistogram.pdf')

res1 = (np.array([np.linalg.norm(x) ** 2 for x in (psffinal - psftru)]) /
               np.array([np.linalg.norm(x) ** 2 for x in psftru]))
plt.hist(res1,bins = 'auto', label = 'final')
plt.xlabel('PSF error histogram')
plt.savefig('final PSF error histogram.pdf')
plt.legend(loc = 'upper right')
plt.show()