#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 17:59:30 2017

@author: pran
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colorbar import make_axes
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
#from string import extract_num


def extract_num(string, format=None):

    numeric = filter(str.isdigit, string)

    if format is int:
        return int(numeric)

    elif format is float:
        return float(numeric)

    else:
        return numeric

def get_images(id, path, suffix):

    files = [path + x for x in os.listdir(path)]

    images = [np.load(file)[id] for file in files if file.endswith(suffix)]

    return np.array(images)


# Set font size and colourmap
fontsize = 18

colors = [(0, 0, 1), (1, 0, 0), (1, 1, 0)]  # B -> R -> Y

cmap = LinearSegmentedColormap.from_list('my_colormap', colors)

# cmap = plt.cm.bwr

cmap.set_under('k')


# Get directory path
obj_id = [1]

# Read in images
clean = data = np.load('Latest_PSFs_Peipei/PSF_tru.npy')[1]

# Make plot

titles = ['S/N = 1.0', 'S/N = 2.0', 'S/N = 3.0', 'S/N = 5.0']
y_labels = ['Obser Image', 'PSFtru\n Deconvolution',  'PSFinterp\n Deconvolution']
#y_labels = ['PSF\n initial PSF_tru', 'PSF\n initial PSF_interp',  'Error\n between two types']
vmin = 0.0005
vmax = np.max(clean)
step = (vmax - vmin) / 6
boundaries = np.arange(vmin, vmax, 0.0001)
ticks = np.arange(vmin, vmax + step, step)
norm = BoundaryNorm(boundaries, plt.cm.get_cmap(name=cmap).N)
plt.figure(1)
im = plt.imshow(clean, norm=norm, cmap=cmap, interpolation='nearest')
plt.title('Clean Image', fontsize=fontsize)
plt.colorbar(im, ticks=ticks)
file_name1 = 'clean_image_' + str(obj_id) + '.pdf'
plt.savefig(file_name1)
print 'Output saved to:', file_name1