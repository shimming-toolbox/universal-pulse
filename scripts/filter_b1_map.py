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
# load images
nii_mask = nib.load('acdc_spine_7t_037_48_T0000_th15.nii.gz')
nii_data = nib.load('acdc_spine_7t_037_56_T0000.nii.gz')
data = nii_data.get_fdata().astype(np.float32)
# convert masked voxels to nan
ind_zero = np.where(nii_mask.get_fdata() == 0.0)
data_nan = data.copy()
data_nan[ind_zero] = np.nan

# The code below is copied from astropy's tutorial and works for 2D but not for 3D. I'm putting it here to help prototyping a new class.
# from astropy.convolution import convolve
# from astropy.convolution import Gaussian2DKernel  # ideally we would create a new class Gaussian3DKernel
# kernel = Gaussian2DKernel(x_stddev=1)
# data_nan_astropy_conv = convolve(data_nan, kernel)

# Save data
nii_data_filt = nib.nifti1.Nifti1Image(data_nan, nii_data.affine)
nib.save(nii_data_filt, 'bla.nii.gz')
