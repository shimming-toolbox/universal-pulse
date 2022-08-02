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
from custom_filters import median_filter

# Chose filter type
filter_type = 'median' # median or gaussian
filter_size = 5 #if median

# load images
nii_mask = nib.load('acdc_spine_7t_037_48_T0000_th15.nii.gz')
nii_data = nib.load('acdc_spine_7t_037_56_T0000.nii.gz')

# 3D Gaussian filtering, ignoring NaN
if filter_type == 'gaussian':
    data = nii_data.get_fdata().astype(np.float32)
    
    # convert masked voxels to nan
    ind_zero = np.where(nii_mask.get_fdata() == 0.0)
    data_nan = data.copy()
    data_nan[ind_zero] = np.nan

    kernel = Gaussian3DKernel(stddev=1)
    data_filt = convolve(data_nan, kernel)
    filename = 'bla_filt_gauss.nii.gz'

    nii_data_filt = nib.nifti1.Nifti1Image(data_filt, nii_data.affine)

elif filter_type == 'median':
    nii_data_filt = median_filter(nii_data, nii_mask, kernel_size=filter_size)
    filename = 'bla_filt_median.nii.gz'

# Save data
nib.save(nii_data_filt, filename)
