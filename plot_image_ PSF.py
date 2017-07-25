#! /usr/bin/env Python

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colorbar import make_axes
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
#from string import extract_num

#from stats import gaussian_kernel
def gaussian_kernel(data_shape, sigma, norm='max'):
    """Gaussian kernel

    This method produces a Gaussian kerenal of a specified size and dispersion

    Parameters
    ----------
    data_shape : tuple
        Desiered shape of the kernel
    sigma : float
        Standard deviation of the kernel
    norm : str {'max', 'sum'}, optional
        Normalisation of the kerenl (options are 'max' or 'sum')

    Returns
    -------
    np.ndarray kernel

    """

    kernel = np.array(Gaussian2DKernel(sigma, x_size=data_shape[1],
                      y_size=data_shape[0]))

    if norm is 'max':
        return kernel / np.max(kernel)

    elif norm is 'sum':
        return kernel / np.sum(kernel)

    else:
        return kernel


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

# colors1 = plt.cm.summer(np.linspace(0, 1, 50))
# colors2 = plt.cm.spring_r(np.linspace(0, 2, 206))
# colors = np.vstack((colors1, colors2))

# colors1 = plt.cm.summer(np.linspace(0, 1, 128))
# colors2 = plt.cm.spring_r(np.linspace(0, 1, 128))
# colors = np.vstack((colors1, colors2))

colors = [(0, 0, 1), (1, 0, 0), (1, 1, 0)]  # B -> R -> Y

cmap = LinearSegmentedColormap.from_list('my_colormap', colors)

# cmap = plt.cm.bwr

cmap.set_under('k')


# Get directory path
obj_id = [200]

# Read in images
clean = np.load('/Users/pran/Documents/code/new/Latest_PSFs_Peipei/PSF_tru.npy')[200]
Initial_RCAPSF = np.load( '/Users/pran/Documents/code/new/Latest_PSFs_Peipei/PSF_RCAcenter.npy')[200]

data_obs = get_images(obj_id, '/Users/pran/Documents/code/new/Latest_PSFs_Peipei/data_obs/', '.npy')
psffinal_perfPSF = get_images(obj_id, '/Users/pran/Documents/code/new/latest_output_v1/PSF_perfcenter/psffinal/', 'psfperf.npy')
psffinal_ExPSF = get_images(obj_id, '/Users/pran/Documents/code/new/latest_output_v1/PSF_Excenter/psffinal/', 'psfEx.npy')
psffinal_RCAPSF = get_images(obj_id, '/Users/pran/Documents/code/new/latest_output_v1/PSF_RCAcenter/psffinal/', 'psfRCA.npy')

data = np.vstack([psffinal_perfPSF, psffinal_ExPSF, psffinal_RCAPSF])
# Make plot

titles = ['S/N = 1.0', 'S/N = 2.0', 'S/N = 3.0', 'S/N = 5.0']
y_labels = ['psffinal_perfPSF', 'psffinal_ExPSF',  'psffinal_RCAPSF']
vmin = 0.0005
vmax = np.max(data)
step = (vmax - vmin) / 6
boundaries = np.arange(vmin, vmax, 0.0001)
ticks = np.arange(vmin, vmax + step, step)
norm = BoundaryNorm(boundaries, plt.cm.get_cmap(name=cmap).N)
n_sigma = data_obs.shape[0]
plt.figure(1)
im = plt.imshow(clean, norm=norm, cmap=cmap, interpolation='nearest')
plt.title('Clean Image', fontsize=fontsize)
plt.colorbar(im, ticks=ticks)
file_name1 = 'clean_psf_' + str(obj_id) + '.pdf'
plt.savefig(file_name1)
print 'Output saved to:', file_name1

plt.figure(2, figsize=(16, 12))
grid = GridSpec(3,n_sigma, wspace=0.05, hspace=0.1)
axes = [plt.subplot(gs) for gs in grid]
for i in range(len(data)):
    im = axes[i].imshow(np.abs(data[i][0]), norm=norm, cmap=cmap,
                        interpolation='nearest')
    if i < len(titles):
        axes[i].set_title(titles[i], fontsize=fontsize)
    if i % n_sigma == 0:
        axes[i].set_ylabel(y_labels[i / n_sigma], fontsize=fontsize)
    else:
        axes[i].get_yaxis().set_visible(False)
    if i < len(data) - n_sigma:
        axes[i].get_xaxis().set_visible(False)
cax, kw = make_axes(axes)
plt.colorbar(im, ticks=ticks, cax=cax, **kw)

file_name2 = 'deconvolved_psfs_' + str(obj_id) + '.pdf'

plt.savefig(file_name2)
print 'Output saved to:', file_name2


