import numpy as np
import numpy.random as npr
from numpy.lib import stride_tricks

import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr

# Set up our R namespaces
R = rpy2.robjects.r
DTW = importr('dtw')

def euclidean(t1, t2, **kwargs):
    return np.sqrt(np.sum(np.abs(t1**2-t2**2)))

def dtw(t1,t2, local_complexity_fcn=None, global_complexity_fcn=None,
        window_size=5, clear_workspace=False, dtw_options={}):
    """ Calculate the dynamic time warping distance between two time series
    arrays.

    When passing in a function using the local_complexity_fcn keyword, the DTW
    cost matrix will be augmented with an additional factor: a metric
    that scales the distance by the difference of complexity between the
    two timeseries objects at those points.

    This is done by creating local complexity vectors for each time series
    object by taking a sliding window and processing it with
    local_complexity_fcn.  However, instead of comparing the complexities 
    directly, each local complexity vector is scaled by the global complexity
    for the time series object. This represents how the local sliding window
    complexity contributes to the overall complexity of the time series. This
    can be thought of normalizing the complexity by the amount of noise within
    the signal. As a result, when using local complexity as a metric for
    similarity search, you can encourage the DTW algorithm to look for similar
    features via local complexity.

    Inputs
    ------

        t1: A numpy array representing the query vector.

        t2: A Numpy array representing the reference vector.

        local_complexity_fcn: A function pointing to the complexity function
        to be used for calculating local complexity.

        window_size: The window size used for local_complexity_fcn.

        clear_workspace: Whether or not the R workspace will be cleared after
        running the function.

        dtw_options: A dictionary that will be passed to the dtw function in R

    Outputs
    -------
    
        distance: The resulting DTW distance between t1 and t2.

        alignment: The R data frame containing additional data. This is
        returned only if clear_workspace=False.

        complexity_vectors: The complexity profiles for each time series
        data input. This is returned only if local_complexity_fcn
        is defined.

    Examples
    --------

    Notes
    -----
        
        1) Right now, it is assumed that t1 and t2 have the same vector
           length. Althought DTW can traditionally handle varying vector
           lengths, the vectors are treated as similar length for
           performance reasons.

        2) The window_size is assumed to be odd. If it is even, it will
           be increased by 1. This is to make sure that the local
           complexity values stay aligned with the corresponding vectors.

        3) The complexity vector that's created for the two time series
           objects are padded and the front and end using a zero-hold
           approach, meaning that for a window_size value of 5, the
           first two elements of the complexity vector are going to have
           the same value as the third element.

    """

    window_size = int(window_size)

    if local_complexity_fcn is not None:

        # We're going to want out window_size to be odd so  that we can ensure
        # that each sliding window slice is centered around a data point.
        window_size += window_size-(window_size%2-1)

        # Calculate the global complexity for each time series object.
        global_complexity = np.array([local_complexity_fcn(t) for t in [t1,t2]])

        # Combine our data into a single multi-dimensional vector
        data = np.array((t1, t2))

        # Generate a collection of subsequences for each time-series
        # component based upon the window size
        if window_size < data.shape[1]:
            subsequences = stride_tricks.as_strided(data,
                                shape=(2, data[0].size-window_size/2*2, window_size),
                                strides=(data.strides[0], data.strides[1], data.strides[1]))
        else:
            window_size = data.shape[1]
            subsequences = np.array([[t1],[t2]])

        # Process our subsequences with the complexity function
        complexity_vectors = np.array(map(lambda x: map(lambda y: local_complexity_fcn(y), x), subsequences))

        # We need to append on values for our complexity vector since it will have
        # less elements due to the sliding window
        complexity_vectors = np.concatenate((
            np.repeat(complexity_vectors[:,0].reshape((2,1)),window_size/2, axis=1)*
                np.ones((2, window_size/2)),
            complexity_vectors,
            np.repeat(complexity_vectors[:,-1].reshape((2,1)),window_size/2, axis=1)*
                np.ones((2, window_size/2))), axis=1)

        # Scale our complexity vectors by the global complexity for
        # each time-series.
        complexity_vectors = np.array([complexity/global_value for complexity,global_value in zip(complexity_vectors,global_complexity)])

        # Calculate a complexity matrix
        T1 = np.repeat(np.matrix(complexity_vectors[0]), complexity_vectors[0].size, axis=0)
        T2 = np.repeat(np.matrix(complexity_vectors[1]).T, complexity_vectors[1].size, axis=1)

        complexity = np.divide(np.maximum(T1,T2), np.minimum(T1,T2))
        complexity = np.where((complexity==0) | (complexity==np.inf) | np.isnan(complexity),1.0,complexity)

        # Create the distance matrix
        distance = np.abs( np.repeat(np.matrix(t1).T, t2.size, axis=1)-
                           np.repeat(np.matrix(t2), t1.size, axis=0))

        # Create a cost matrix using both complexity and distance matrices
        cost = [np.array(np.multiply(complexity, distance))]

    else:
        cost = [t1, t2]

    # Calculate the DTW
    dtw_options.update({'keep.internals': True})

    # Correctly adjust the window.size attribute. A user may pass in either a
    # percentage of the dataset to be used in the window size.
    if 'window.size' in dtw_options.keys():
        if dtw_options['window.size'] > 0.0 and \
           dtw_options['window.size'] <= 1.0 and \
           type(dtw_options['window.size']) == float:
               dtw_options['window.size'] = int(round(dtw_options['window.size'] * t1.size))

    try:
        alignment = R.dtw(*cost, **dtw_options)
    except:
        return None

    # Capture the normalized distance.
    distance = alignment.rx('normalizedDistance')[0][0]

    # Scale the distance by a global complexity function if need be
    if global_complexity_fcn is not None:
        distance *= global_complexity_fcn(t1,t2)
    
    # Return the results.
    if clear_workspace:
	    # Clear our workspace in R.
        R('rm(list = ls(all = TRUE))')
        return distance
    elif local_complexity_fcn is not None:
	    # Return the distance, alignment, and complexity vectors
        return distance,alignment,complexity_vectors
    else:
        return distance,alignment
