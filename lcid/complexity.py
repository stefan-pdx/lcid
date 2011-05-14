import numpy as np

def simple_complexity(data):
    """ Calculate the complexity of a series of data by calculating the
    length of the square root of the sume of the squares of the
    v-yalues for the signal.

    Inputs
    ------
    
        data: A Numpy array-like object

    Output
    ------

        complexity: A float value representing the complexity of the data.

    Examples
    --------

    """
    return np.sqrt(np.sum(np.diff(data)**2))

def complexity_correction_factor(t1, t2):
    complexities = [simple_complexity(t) for t in [t1,t2]]
    return max(complexities)/min(complexities)
