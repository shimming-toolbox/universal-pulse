import numpy as np
import scipy.ndimage as spi
import nibabel as nib


def median_func(x):
    """ Median func
    Objective function for median filter

    At edges defined by NaNs (eg, from mask), values outside the volume
    are not counted in the median. This is to allow for better edge
    preservation.
    """

    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    else:
        x = np.median(x)
        return x


def median_filter(img, mask=None, kernel_size=5):
    """ Median filter for NIFTI volume.
    Application: B1+ map smoothing.

    img: NIFTI volume to be smoothed, in a Nibabel object.
    mask: NIFTI volume to mask img, in a Nibabel object.
    kernel_size: number of voxels to be used for median kernel for the in-plane smoothing.
        Through-slice smoothing will be adjusted to match the same physical distance if
        slice thickness is different from in-plane resolution.

    return: Filtered and masked img.
    """

    data = img.get_fdata().astype(np.float32)

    data_mask = data.copy()

    # Mask image with NaNs
    if mask is not None:
        # Convert masked voxels to nan
        ind_zero = np.where(mask.get_fdata() == 0.0)
        data_mask[ind_zero] = np.nan

    # Get image voxel dim info
    pixdim = img.header["pixdim"][1:4]

    # Setup filter "footprint", ndarray where median is going to be calculated around the pixel
    # of interest. Round up to integers when calculating array dims.
    norm_dim = pixdim / min(pixdim)
    kernel_size = np.ndarray.astype(np.ceil(kernel_size / (norm_dim)), dtype=int)
    footprint = np.ones(kernel_size)

    data_filt = spi.generic_filter(data_mask, median_func, footprint=footprint)

    if mask is not None:
        data_filt[ind_zero] = np.nan

    img_filt = nib.Nifti1Image(data_filt, img.affine, img.header)

    return img_filt
