'''
tmd class : Soma
'''


class Soma(object):
    '''Class of neuron soma
    '''
    import numpy as _np
    from tmd.Soma.methods import get_center
    from tmd.Soma.methods import get_diameter

    def __init__(self, x=_np.array([]), y=_np.array([]), z=_np.array([]),
                 d=_np.array([])):
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
        import numpy as np

        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)
        self.z = np.array(z, dtype=float)
        self.d = np.array(d, dtype=float)

    def copy_soma(self):
        """
        Returns a deep copy of the Soma.
        """
        import copy
        return copy.deepcopy(self)

    def is_equal(self, soma):
        '''Tests if all soma data are the same'''
        import numpy as np

        eq = np.alltrue([np.allclose(self.x, soma.x, atol=1e-4),
                         np.allclose(self.y, soma.y, atol=1e-4),
                         np.allclose(self.z, soma.z, atol=1e-4),
                         np.allclose(self.d, soma.d, atol=1e-4)])
        return eq
