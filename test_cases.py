""" This script is used to define a class that can be """
import pdb

import sys, os
sys.path.append(os.path.join(os.getcwd(), '../'))

import pylab
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import lcid.distance
import lcid.complexity

data_dir = lambda x: os.path.join(os.getcwd(), '../data/%s'%x)
algorithm_configs = {'cid': {'global_complexity_fcn': lcid.complexity.complexity_correction_factor},
                     'lcid': {'local_complexity_fcn': lcid.complexity.simple_complexity}}

class SpecialCase(object):
    """ A SpecialCase class which represents a data set of interest.

    Examples
    --------

        >>> import test_cases
        >>> Coffee = SpecialCase('Coffee', window_size=0.15,
        ...              dtw_window_size=0.03, case_type='+')
        >>> Coffee.plot() #Creates Coffee.png
        
    """
    name = ''
    data = {}
    max_distance = None

    dtw_options = {'window.type': 'slantedband',
                   'window.size': 5}

    def __init__(self, name, window_size, dtw_window_size, case_type):
        """ Constructor for the SpecialCase class.

        Inputs
        ------

            name: Name of the dataset that corresponds to the file name.

            window_size: The window size used in the local complexity function
                sliding window. 

            dtw_window_size: The size of the window used for the DTW call.

            case_type: A string representing the case type. Should be either
                '+' or '-', representing if the LCID appoach did better or
                worse than the CID implementation.

        """

        self.name = name

        # Load in the data
        for data_type in ['test','train']:
            self.data[data_type] = np.loadtxt(os.path.join(data_dir(name),
                '%s_%s'%(name,data_type.upper())))
        
        self.window_size = window_size * (self.data['test'][0].size-1)
        self.dtw_options['window.size'] = dtw_window_size * \
                (self.data['test'][0].size-1)

        # Select the right algorithm config.
        if case_type == '+':
            correct_algorithm_config = algorithm_configs['lcid']
            incorrect_algorithm_config = algorithm_configs['cid']
        elif case_type =='-':
            correct_algorithm_config = algorithm_configs['cid']
            incorrect_algorithm_config = algorithm_configs['lcid']

        max_distance = (-1, -1, -1, 0)
        
        # Select the test case of interest.
        for test_id, test_case in enumerate(self.data['test']):
            good_cases = []
            bad_cases = []

            for train_id, train_case in enumerate(self.data['train']):

                good_cases.append(lcid.distance.dtw(test_case[1:], 
                    train_case[1:], clear_workspace=True,
                    window_size=self.window_size, 
                    dtw_options=self.dtw_options,
                    **correct_algorithm_config))

                bad_cases.append(lcid.distance.dtw(test_case[1:],
                    train_case[1:], clear_workspace=True,
                    window_size=self.window_size,
                    dtw_options=self.dtw_options,
                    **incorrect_algorithm_config))

            # We should now have two distances for the good and bad case.
            # We want to capture the case where the good case made the right
            # call and the bad case made the incorrect call.
            good_id = np.argmin(good_cases)
            bad_id = np.argmin(bad_cases)
            if test_case[0] == self.data['train'][good_id][0] and \
               test_case[0] != self.data['train'][bad_id][0]:
                   if bad_cases[good_id]-good_cases[good_id] > max_distance[3]:
                       max_distance = (test_id, good_id, bad_id,
                               bad_cases[good_id]-good_cases[good_id])

        self.max_distance = max_distance

        # Recalculate the alignment matrices for the two cases
        self.good_case = lcid.distance.dtw(
                self.data['test'][max_distance[0],1:],
                self.data['train'][max_distance[1],1:],
                clear_workspace=False, window_size=self.window_size,
                dtw_options=self.dtw_options, **correct_algorithm_config)
        
        self.bad_case = lcid.distance.dtw(
                self.data['test'][max_distance[0],1:],
                self.data['train'][max_distance[2],1:],
                clear_workspace=False, window_size=self.window_size,
                dtw_options=self.dtw_options, **incorrect_algorithm_config)

    def plot(self):
        """ Plot out the DTW alignment plots for the given case. The plot
        is automatically saved to <self.name>.png """

        # Plot the test case and the two training cases referenced as closest.
        X = 4*[np.linspace(0, 1, self.data['test'][self.max_distance[0]].size)]
        Y = [self.data['train'][self.max_distance[1]],
             self.data['test'][self.max_distance[0]],
             self.data['train'][self.max_distance[1]],
             self.data['test'][self.max_distance[0]]]
        C = [self.data['train'][self.max_distance[1]],
             self.data['test'][self.max_distance[0]],
             self.data['train'][self.max_distance[1]],
             self.data['test'][self.max_distance[0]]]
        colors = ['#4DDE00','#0D56A6','#FF5900','#0D56A6']

        # Scale each dataset to be between 0 and 0.25
        Y = [(y-np.min(y))/np.max(y-np.min(y))*0.25 for y in Y]

        # Shift the four Y's so that each is in a 1/4 partition
        Y = [y+offset for offset,y in zip([idx/4.0 for idx in 
                range(4)[::-1]], Y)]

        # Plot out the warping paths
        indices_1 = np.array(self.good_case[1].rx('index1'), dtype=int)-1
        indices_2 = np.array(self.good_case[1].rx('index2'), dtype=int)-1

        for idx1,idx2 in zip(indices_1[0], indices_2[0]):
            pylab.plot( [X[1][idx1], X[0][idx2]], [Y[1][idx1], Y[0][idx2]],
                    color='#AAAAAA')

        indices_1 = np.array(self.bad_case[1].rx('index1'), dtype=int)-1
        indices_2 = np.array(self.bad_case[1].rx('index2'), dtype=int)-1

        for idx1,idx2 in zip(indices_1[0], indices_2[0]):
            pylab.plot( [X[3][idx1], X[2][idx2]], [Y[3][idx1], Y[2][idx2]],
                    color='#AAAAAA')

        # Plot out the time series data
        plot_data = [None]*12
        plot_data[::3] = X
        plot_data[1::3] = Y
        plot_data[2::3] = colors

        pylab.plot(*plot_data)
        pylab.gca().set_xticks([])
        pylab.gca().set_yticks([])

        pylab.title('%s Dataset: Test Instance Classification'%(self.name))

        pylab.savefig('%s.pdf'%self.name) 
