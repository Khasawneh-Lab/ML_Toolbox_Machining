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
It assumes that the reconstructed Wavelet packets and frequency domain features are available an they are in the same folder with the data files.
It computes feature matrices for training and test set and performs the classification with chosen algorithm. 
It returns the results in an array. It also prints the total elapsed time.
		
.. automodule:: WPT_EEMD_ML.WPT_Transfer_Learning
    :members:
	
Transfer Learning On Two Cases Using WPT
----------------------------------------

This fuction implement transfer learning by training a classifier on two different 
data sets and testing it on remaining two different data sets. Stickout lengths of 
each data set should be determined by user. It asks for file paths to training and test set files. 
It assumes that reconstructed wavelet packets and frequency domain features are available in the specified data file folder.
Algorithm performs classfication based on chosen classifier and returns the results in an array. 
It also prints the total elapsed time.  

.. automodule:: WPT_EEMD_ML.WPT_Transfer_Learning_2case.WPT_Transfer_Learning_2case
	:members:
