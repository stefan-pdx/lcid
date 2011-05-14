""" This script will scrape through all of the .log files in the current
    directory, extract out the results, and format them into a table
    form such that it can be copied-and-pasted into Office, Google Docs, and
    so on. """

import os

files = [file_name for file_name in os.listdir(os.curdir)
            if '.log' in file_name]

results = {}
classification_names = set()
dataset_names = set()

for file_name in files:
    results.update({file_name: {}})

    f = open(file_name)

    for line in f:
        if '|' not in line:
            continue
        
        time_info,results_info = line.split('|')
        dataset = time_info.strip().split()[-1]
        classification_type,error_rate = results_info.strip().split(':')

        if dataset not in results[file_name].keys():
            results[file_name][dataset] = {}

        results[file_name][dataset].update({classification_type: error_rate})

        classification_names.add(classification_type)
        dataset_names.add(dataset)

# Sort the data sets and classification names
classification_names = sorted(list(classification_names))
dataset_names = sorted(list(dataset_names))

for file_name, result in results.iteritems():
    print '-' * len(file_name)
    print file_name
    print '-' * len(file_name)
    print '\t'.join(['Dataset'] + classification_names)

    for dataset in dataset_names:

        print '\t'.join([dataset] + [result[dataset][classification_type] for
            classification_type in classification_names if
            classification_type in result[dataset].keys()])
