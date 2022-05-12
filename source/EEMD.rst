.. _EEMD:


Ensemble Empirical Mode Decomposition (EEMD)
============================================

.. automodule:: WPT_EEMD_ML.EEMD_Feature_Extraction
    :members:
.. automodule:: WPT_EEMD_ML.EEMD_Transfer_Learning
    :members:

::

	# Compute IMFs

	from EEMD_Feature_Extraction import EEMD_IMF_Compute
	data_path = 'D:\\Data Archive\\Cutting_Test_Data_Documented\\cutting_tests_processed\\2inch_stickout'
	list_name = 'time_series_name_2inch.txt' 
	label_name = '2_inch_Labels_2Class.npy'
	EEMDecs = 'A'
	saving = True
	saving_path = 'D:\\Repositories\\WPT_EEMD_ML_Machining\\test\\EEMD_Output'

	infoEMF,split_labels = EEMD_IMF_Compute(data_path,list_name, EEMDecs, saving, saving_path)


	# feature extraction
	from EEMD_Feature_Extraction import EEMD_Feature_Compute
	p=2
	feature_mat = EEMD_Feature_Compute(infoEMF,p)

	# classification
	from EEMD_Classification import EEEMD_Classification
	cv = 5
	labels = split_labels[:,2]
	saving = True
	param_tuning = False
	feature_ranking = False
	saving_path = 'D:\Repositories\WPT_EEMD_ML_Machining\\test\EEMD_Output'

	clas_rep_test,clas_rep_train = EEEMD_Classification(feature_mat,cv, labels,param_tuning, feature_ranking, saving, saving_path)

	#plot the results
	from Plot_Results import plot_results
	import numpy as np

	methods = ['EEMD']  
	clsf_names = ['SVM','LR','RF','GB']
	cv = 5
	layout = [1,1]
	ylabel_index = np.array([1])
	res_path = 'D:\\Repositories\\WPT_EEMD_ML_Machining\\test\\EEMD_Output\\'
	n_feature = 7

	# plot the results
	fig = plot_results(res_path,param_tuning,feature_ranking,n_feature,methods,clsf_names,cv,layout,ylabel_index)




	# ------------------------------------------------------------------------------

	# EEMD Transfer Learning

	from WPT_EEMD_ML.EEMD_Transfer_Learning import EEMD_Transfer_Learning
	from Plot_Results import plot_results
	import numpy as np


	data_paths = []
	data_paths.append('D:\\Data Archive\\Cutting_Test_Data_Documented\\cutting_tests_processed\\2inch_stickout')
	data_paths.append('D:\\Data Archive\\Cutting_Test_Data_Documented\\cutting_tests_processed\\4p5inch_stickout')

	list_names = []
	list_names.append('time_series_name_2inch.txt')
	list_names.append('time_series_name_4p5inch.txt')

	Decomps=[]
	Decomps.append('NA')
	Decomps.append('NA')

	info_IMFs = []
	info_IMFs.append(2)
	info_IMFs.append(1)

	saving = True
	param_tuning=False
	feature_ranking = False
	cv=5

	saving_path = 'D:\Repositories\WPT_EEMD_ML_Machining\\test\EEMD_Output'

	EEMD_Transfer_Learning(data_paths,list_names,Decomps, info_IMFs, cv,param_tuning,feature_ranking,saving,saving_path)


	# plot the results
	   
	methods = ['EEMD']  
	clsf_names = ['SVM','LR','RF','GB']
	cv = 5
	layout = [1,1]
	ylabel_index = np.array([1])
	res_path = 'D:\\Repositories\\WPT_EEMD_ML_Machining\\test\\EEMD_Output\\'
	n_feature = 7

	fig = plot_results(res_path,param_tuning,feature_ranking,n_feature,methods,clsf_names,cv,layout,ylabel_index)
