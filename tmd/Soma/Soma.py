'''
tmd class : Soma
'''
import copy
import numpy as np


class Soma(object):
    '''Class of neuron soma
    '''
    # pylint: disable=import-outside-toplevel
    from tmd.Soma.methods import get_center
    from tmd.Soma.methods import get_diameter

    def __init__(self, x=np.array([]), y=np.array([]), z=np.array([]),
                 d=np.array([])):
        """
        Constructor for tmd Soma Object

        Parameters
        ----------
        x : numpy array
            The x-coordinates of surface trace of neuron soma.
        y : numpy array
            The y-coordinates of surface trace of neuron soma.
        z : numpy array
            The z-coordinate of surface trace of neuron soma.
        d : numpy array
            The diameters of surface trace of neuron soma.
        ----------
        """
        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)
        self.z = np.array(z, dtype=float)
        self.d = np.array(d, dtype=float)

    def copy_soma(self):
        """
        Returns a deep copy of the Soma.
        """
        return copy.deepcopy(self)

    def is_equal(self, soma):
        '''Tests if all soma data are the same'''
        eq = np.alltrue([np.allclose(self.x, soma.x, atol=1e-4),
                         np.allclose(self.y, soma.y, atol=1e-4),
                         np.allclose(self.z, soma.z, atol=1e-4),
                         np.allclose(self.d, soma.d, atol=1e-4)])
        return eq
