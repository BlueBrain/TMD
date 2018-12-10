'''
tmd transformation algorithms
'''
import numpy as np


def tmd_scale(barcode, thickness):
    '''Only the first two components will be scaled according to
       the thickness parameter, because they correspond to
       spatial coordinates
    '''
    scaling_factor = np.ones(len(barcode[0]), dtype=np.float)
    scaling_factor[:2] = thickness
    return np.multiply(barcode, scaling_factor).tolist()
