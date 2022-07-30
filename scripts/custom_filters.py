import numpy as np

def median_func(x):
    x = x[~np.isnan(x)]
    if x.size==0:
        return np.nan 
    else:
        x=np.median(x)
        return x