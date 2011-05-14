Local Complexity-Invariant Distance Analysis
============================================

This project contains the code, data, and results to explore the use of a local complexity-invariant distance measure with dynamic time warping. The paper regarding this study can be foun [here](http://web.cos.gmu.edu/~snovak/Exploring%20A%20Local%20Complexity-Invariant%20Distance%20Measure%20with%20Dynamic%20Time%20Warping.pdf).

Requirements
------------

This study requires the following software configuration:

* [Python](http://www.python.org). Python 2.7 was used for this study, but you could probably get away with using 2.6+.
* [Numpy](http://numpy.scipy.org/).
* [R](http://www.r-project.org/).
* [rpy2](http://rpy.sourceforge.net/rpy2.html). This is a Python-to-R interface that we use to bring in some additional computational functionality.
* [Matplotlib](http://matplotlib.sourceforge.net/). Only required if you want to generate the warping path visualization graphs.

Additionally, there is suppose for [Growl](http://growl.info/) for those using OS X, like myself. In this case we get desktop notifications for events from the analysis. It's not required and the code will run fine without it.

Overview
--------

Take a look at the `lcid` module, which contains to submodules: `complexity` and `distance`. Each of these contains functionality to calculate the the corresponding complexity and distance for multiple time series data.

The main script, `analysis.py` uses the `lcid` module to process 13 data sets which can be found in the `data` folder. This file should be able to run as-is, as long as you meet the above requirements. Furthermore, the `test_cases.py` file allows you to select a specific case and visualize the warping path differences between the global and local interpretations of a complexity-invariant distance metric.
