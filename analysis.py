import os, sys, logging
from multiprocessing import Pool
import numpy as np
import itertools
import time
from operator import itemgetter

# Set up Growl bindings if available. This will give us desktop notifications
# on the status of the analysis.
try:
    import Growl
    notifier = Growl.GrowlNotifier('LCID Analysis',
        ['Analysis Status Update'])
    notifier.register()

    def notify(msg):
        notifier.notify('Analysis Status Update', 'Status Update',
            '%s @ %s'%(msg, time.asctime(time.localtime(time.time()))),
            sticky=True)

except ImportError:

    def notify(msg):
        pass

# Import our algorithm. Note that the import statement is assuming
# the LCID folder containing complexity.py and distance.py is
# located in the same folder as this script.
sys.path.append(os.getcwd())
import lcid.distance
import lcid.complexity

# Set up our logger
logger = logging.getLogger('lcid')
hdlr = logging.FileHandler('analysis.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

# Load datasets - note that we have the sorted in ascending order
# to dataset size so that we can process the smaller ones first.
# We will be selecting the datasets of interest later on.
datasets = [{'name': 'Coffee',                 'dtw_window_size': 0.03},
            {'name': 'Beef',                   'dtw_window_size': 0.00},
            {'name': 'OliveOil',               'dtw_window_size': 0.01},
            {'name': 'Lighting2',              'dtw_window_size': 0.06},
            {'name': 'Lighting7',              'dtw_window_size': 0.05},
            {'name': 'FaceFour',               'dtw_window_size': 0.02},
            {'name': 'ECG200',                 'dtw_window_size': 0.00},
            {'name': 'Trace',                  'dtw_window_size': 0.03},
            {'name': 'Gun_Point',              'dtw_window_size': 0.00},
            {'name': 'Fish',                   'dtw_window_size': 0.04},
            {'name': 'OSULeaf',                'dtw_window_size': 0.07},
            {'name': 'Synthetic_control',      'dtw_window_size': 0.06},
            {'name': 'DiatomSizeReduction',    'dtw_window_size': 0.00},
            {'name': 'Haptics',                'dtw_window_size': 0.02},
            {'name': 'Cricket_X',              'dtw_window_size': 0.07},
            {'name': 'Cricket_Y',              'dtw_window_size': 0.17},
            {'name': 'Cricket_Z',              'dtw_window_size': 0.07},
            {'name': 'Adiac',                  'dtw_window_size': 0.03},
            {'name': '50words',                'dtw_window_size': 0.06},
            {'name': 'InlineSkate',            'dtw_window_size': 0.14},
            {'name': 'SonyAIBORobotSurface',   'dtw_window_size': 0.00},
            {'name': 'SwedishLeaf',            'dtw_window_size': 0.02},
            {'name': 'WordsSynonyms',          'dtw_window_size': 0.09},
            {'name': 'MedicalImages',          'dtw_window_size': 0.20},
            {'name': 'ECGFiveDays',            'dtw_window_size': 0.00},
            {'name': 'CBF',                    'dtw_window_size': 0.11},
            {'name': 'SonyAIBORobotSurfaceII', 'dtw_window_size': 0.00},
            {'name': 'Symbols',                'dtw_window_size': 0.08},
            {'name': 'ItalyPowerDemand',       'dtw_window_size': 0.00},
            {'name': 'TwoLeadECG',             'dtw_window_size': 0.04},
            {'name': 'Motes',                  'dtw_window_size': 0.01},
            {'name': 'CinC_ECG_torso',         'dtw_window_size': 0.01},
            {'name': 'FaceAll',                'dtw_window_size': 0.03},
            {'name': 'FacesUCR',               'dtw_window_size': 0.12},
            {'name': 'MALLAT',                 'dtw_window_size': 0.00},
            {'name': 'Yoga',                   'dtw_window_size': 0.02},
            {'name': 'uWaveGestureLibrary_X',  'dtw_window_size': 0.04},
            {'name': 'uWaveGestureLibrary_Y',  'dtw_window_size': 0.04},
            {'name': 'uWaveGestureLibrary_Z',  'dtw_window_size': 0.06},
            {'name': 'ChlorineConcentration',  'dtw_window_size': 0.00},
            {'name': 'Two_Patterns',           'dtw_window_size': 0.04},
            {'name': 'Wafer',                  'dtw_window_size': 0.01},
            {'name': 'StarLightCurves',        'dtw_window_size': 0.06}]

# Collect our training and test data files
data_dir = lambda x: os.getcwd() + '/data/%s'%x

for dataset in datasets:
    dataset.update({
        'training_file': os.path.join(data_dir(dataset['name']),
                         '%s_TRAIN'%dataset['name']),
        'test_file' : os.path.join(data_dir(dataset['name']),
                         '%s_TEST'%dataset['name'])})

# Set up classification routines used
classification_methods = [
        {
        'name':     '1-NN Euclidean',
        'function': lcid.distance.euclidean,
        'options':  lambda dataset: {}
        },{
        'name':     '1-NN Best Warping Window DTW',
        'function': lcid.distance.dtw, 
        'options':  lambda dataset: {
            'clear_workspace': True,
            'dtw_options': {
                'window.type': 'slantedband', 
                'window.size': dataset['dtw_window_size']}}
        },{
        'name':     '1-NN No Warping Window',
        'function': lcid.distance.dtw, 
        'options':  lambda dataset: {
            'clear_workspace': True}
        },{
        'name':     '1-NN Complexity Invariant DTW',
        'function': lcid.distance.dtw, 
        'options':  lambda dataset: {
            'global_complexity_fcn': 
                lcid.complexity.complexity_correction_factor,
            'clear_workspace': True,
            'dtw_options': {
                'window.type': 'slantedband', 
                'window.size': dataset['dtw_window_size']}}
        },{
        'name':     '1-NN Local Complexity Invariant DTW',
        'function': lcid.distance.dtw, 
        'options':  lambda dataset: {
            'local_complexity_fcn': lcid.complexity.simple_complexity,
            'clear_workspace': True,
            'dtw_options': {
                'window.type': 'slantedband', 
                'window.size': dataset['dtw_window_size']}}
        },{
        'name':     '1-NN Local and Global Complexity Invariant DTW',
        'function': lcid.distance.dtw, 
        'options':  lambda dataset: {
            'local_complexity_fcn': lcid.complexity.simple_complexity,
            'global_complexity_fcn':
                 lcid.complexity.complexity_correction_factor,
            'clear_workspace': True,
            'dtw_options': {
                'window.type': 'slantedband', 
                'window.size': dataset['dtw_window_size']}}}]

def process_test_case(inputs):

    # Extract out the inputs
    (classification_id, test_case, training_data,
            test_data, window_size_factor) = inputs
    classification_method = classification_methods[classification_id]

    # Calculate the distane to each training case
    distances = []
    for training_case in training_data:
        distances.append(classification_method['function'](
            test_case[1:], training_case[1:],
            window_size=int(window_size_factor*test_data.shape[1]),
            **classification_method['options'](dataset)))

    # Extract out the minimum distance and return 0/1 for fail/pass
    min_ind = np.argmin([distance[0] for distance in distances] if 
            type(distances[0])==tuple else distances)
    return int(int(test_case[0]) == int(training_data[min_ind,0]))

if __name__ == '__main__':

    # Configure to use the multiprocessing module. If enabled, there will be a
    # significant increase in performance, since we can distribute each test
    # case out across active threads. However, for debugging purposes, this
    # should be set to false, otherwise you don't get a very good debug report
    # if the code fails in the process_test_case function.
    use_multiprocessing = True
    testing_mode = False

    if use_multiprocessing:
        pool = Pool(8)

    # We're only going to consider the first 13 datasets
    datasets = datasets[:13]

    # Create a list of classification methods to use. This corresponds to the
    # indicies for classification_methods. This allows us to process the
    # analysis using just a few algorithms to do some testing. Note that in
    # this configuration, we're only going to test the classification methods
    # that utilize LCID. If we wanted to process all of the classification
    # methods, we would change the selected_methods to be range(6).
    selected_methods = [4,5] #range(6)

    # Selected a window size that is used for the sliding window in which the
    # local complexity function is applied to.
    window_size_factor = 0.025

    # Step through each dataset in our analysis.
    for dataset in datasets:

        # Load in the datasets.
        training_data = np.loadtxt(dataset['training_file'])
        test_data = np.loadtxt(dataset['test_file'])

        logger.info('Starting analysis of %s dataset'%(dataset['name']))

        # Step through each classification method and calculate the
        # corresponding error rate.
        for classification_id, classification_method in zip(selected_methods,
                itemgetter(*selected_methods)(classification_methods)):

            # Throw out our test data to the process_test_case function.
            if use_multiprocessing:
                results = pool.map_async(process_test_case, 
                        [(classification_id, test_case, training_data,
                            test_data, window_size_factor) for test_case
                            in test_data])
                results = results.get()
            else:
                results = map(process_test_case, [(classification_id, 
                    test_case, training_data, test_data, window_size_factor)
                    for test_case in test_data])

            # Remove 'None' cases in case there was a failure in caluclating
            # distance.
            results = [result for result in results if results is not None]

            # Log the corresponding error rate.
            error_rate = (1 - sum(results)/float(test_data.shape[0]))*100 \
                if len(results) > 0 else -1.0
            status_message = '%s | %s error rate: %f%%'%(dataset['name'], 
                    classification_methods[classification_id]['name'],
                    error_rate)
            logger.info(status_message)

            if testing_mode:
                notify(status_message)
