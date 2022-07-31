#!/usr/bin/env python
#
# Filter B1 map.
# 
# B1 maps are based on the B1+ interferometry sequence from Neurospin/CEA.
#
# ------------------------------------------------------------------------------
# Copyright (c) 2022 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
# Licence: https://github.com/shimming-toolbox/universal-pulse/blob/main/LICENSE


import numpy as np
import nibabel as nib
from astropy.convolution import convolve
import scipy.ndimage as spi

from gaussian3d import Gaussian3DKernel
from custom_filters import median_func

# Chose filter type
filter_type = 'median' # median or gaussian
filter_size = 5 #if median

# load images
nii_mask = nib.load('acdc_spine_7t_037_48_T0000_th15.nii.gz')
nii_data = nib.load('acdc_spine_7t_037_56_T0000.nii.gz')
data = nii_data.get_fdata().astype(np.float32)
# convert masked voxels to nan
ind_zero = np.where(nii_mask.get_fdata() == 0.0)
data_nan = data.copy()
data_nan[ind_zero] = np.nan

# 3D Gaussian filtering, ignoring NaN
if filter_type == 'gaussian':
    kernel = Gaussian3DKernel(stddev=1)
    data_filt = convolve(data_nan, kernel)
    filename = 'bla_filt_gauss.nii.gz'
elif filter_type == 'median':
    # Get image info
    dim = np.array(nii_data.header['dim'][1:4])
    pixdim = nii_data.header['pixdim'][1:4]

    # Setup filter "footprint", ndarray where median is going to be calculated around the pixel
    # of interest. Round up to integers when calculating array dims.
    norm_dim = pixdim/min(pixdim)
    kernel_size = np.ndarray.astype(np.ceil(filter_size/(norm_dim)), dtype=int)
    footprint= np.ones(kernel_size) 

    data_filt=spi.generic_filter(data_nan, median_func, footprint=footprint)
    filename = 'bla_filt_median.nii.gz'

# Mask filtered map
data_filt[ind_zero] = np.nan

# Save data
nii_data_filt = nib.nifti1.Nifti1Image(data_filt, nii_data.affine)
nib.save(nii_data_filt, filename)
