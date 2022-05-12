.. _WPT:

Wavelet Packet Transform
======================================

Decision Making about Informative Wavelet Packets 
-------------------------------------------------
.. automodule:: WPT_EEMD_ML.WP_Energy_Ratio
    :members:

   
Feature extraction and supervised classification using WPT 
----------------------------------------------------------

This algorithm takes time series for turning experiments as input and it generates the feature matrix based on specified WPT level. 
The reconstructed Wavelet packets and corresponding frequency domain feature matrices should be computed before running this algorithm. Please see the instructions for Matlab functions on this 
documentation before using this algorithm.
The Wavelet packets frequency domain features and datafiles should be in the same folder.
It asks for file paths for the data files.
Algorithm performs the classfication with chosen algorithm and provides results in a np.array.
It plots the mean accuracies and deviations for test and training set with respect to number of features used in classification, if user enables plotting option of the algorithm.
It also prints the total elapsed time. 

.. automodule:: WPT_EEMD_ML.WPT_Feature_Extraction
    :members:
	
Transfer Learning Application Using WPT
---------------------------------------

This function uses transfer learning principles to transfer the knowledge obtained from one cutting configuration to another one.
		
.. automodule:: WPT_EEMD_ML.WPT_Transfer_Learning
    :members:
	

