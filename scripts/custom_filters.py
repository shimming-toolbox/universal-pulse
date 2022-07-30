import numpy as np

def median_func(x):
    ''' Median func
    Objective function for median filter

    At edges defined by NaNs (eg, from mask), values outside the volume
    are not counted in the median. This is to allow for better edge
    preservation.
    '''
    
    x = x[~np.isnan(x)]
    if x.size==0:
        return np.nan 
    else:
        x=np.median(x)
        return x