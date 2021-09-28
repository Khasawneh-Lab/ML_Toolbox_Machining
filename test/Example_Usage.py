
"""
This code provides an example for each function for feature extraction using WPT approach.

"""

# WPT Informative Wavelet Packet Decision Making # ----------------------------
# parameters 
from WPT_EEMD_ML.WP_Energy_Ratio import WP_Energy_Ratio
data_path = 'D:\\Data Archive\\Cutting_Test_Data_Documented\\cutting_tests_processed\\2inch_stickout'
list_name = 'time_series_name_2inch.txt'
ts_no = [1,13,21]
WF = 'db10'
L=4
case_no=2
layout=[4,4]
plot_recon = True
fs= 10000

WP_Energy_Ratio(data_path, list_name, ts_no, WF, L, case_no, fs, plot_recon,layout)


# WPT Reconstruction of Signals from Informative Wavelet Packets # ------------
from WPT_EEMD_ML.WPT_Informative_Packet_Recon import WPT_Informative_Packet_Recon
data_path = 'D:\\Data Archive\\Cutting_Test_Data_Documented\\cutting_tests_processed\\2inch_stickout'
list_name = 'time_series_name_2inch.txt'
label_name = '2_inch_Labels_2Class.npy'
WF = 'db10'
L=4
IWP = 3
# if saving is true, user needs to provide the name of the file which will contain the reconstructed signals
saving = True
recon = WPT_Informative_Packet_Recon(data_path,list_name,WF,L,IWP,saving,'WPT_Output')
# if saving is false, no additional parameter is needed
saving = False
recon = WPT_Informative_Packet_Recon(data_path,list_name,WF,L,IWP,saving)


# WPT Feature Extraction# -----------------------------------------------------
#inputs 
from WPT_EEMD_ML.WPT_Feature_Extraction import WPT_Feature_Extraction
data_path = 'D:\\Data Archive\\Cutting_Test_Data_Documented\\cutting_tests_processed\\4p5inch_stickout'
list_name = 'time_series_name_4p5inch.txt'
WF = 'db10'
L=4
IWP = 10
label_name = '4p5_inch_Labels_2Class.npy'
fs = 10000  
# if saving is false, there is no need for additional parameters
saving = False
feature_mat,labels = WPT_Feature_Extraction(data_path, list_name,label_name,WF,L,IWP,fs,saving)
# if saving is correct, user needs to provide the name of the file which will contain features and the labels
saving = True
feature_mat,labels = WPT_Feature_Extraction(data_path, list_name,label_name,WF,L,IWP,fs,saving,'features')


# WPT Classification ----------------------------------------------------------

from WPT_EEMD_ML.WPT_Classification import WPT_Classification

# inputs
data_path = 'D:\\Data Archive\\Cutting_Test_Data_Documented\\cutting_tests_processed\\2inch_stickout'
list_name = 'time_series_name_2inch.txt'
WF = 'db10'
L=4
IWP = 3
label_name = '2_inch_Labels_2Class.npy'
saving = True
fs = 10000 
param_tuning = False
feature_ranking = False
cv = 5
saving_path = 'D:\Repositories\WPT_EEMD_ML_Machining\\test\WPT_Output'
save_name = "Reconstructions"
reports = WPT_Classification(data_path,list_name,label_name,WF,L,IWP,fs,cv,param_tuning,feature_ranking,saving,saving_path,save_name)


# WPT 2-class classification
import numpy as np 
from WPT_Feature_Extraction import WPT_Feature_Extraction
stickout_length='4p5'
WPT_Level = 3
Classifier = 'SVC'
plotting = False

results = WPT_Feature_Extraction(stickout_length, WPT_Level, 
                                  Classifier, plotting)


# ------------------------------------------------------------------------------

# WPT transfer learning

# add data paths into a list 
# first add the paths for training data sets ,then the ones for test cases
# algorithm will assume the first half of the list is paths for training cases
# and treat other half of the list as the paths for test cases.
# This is also applicable to other variables such as wavelet functions, 
# time series names etc.
from WPT_EEMD_ML.WPT_Transfer_Learning import WPT_Transfer_Learning


data_paths = []
data_paths.append('D:\\Data Archive\\Cutting_Test_Data_Documented\\cutting_tests_processed\\4p5inch_stickout')
data_paths.append('D:\\Data Archive\\Cutting_Test_Data_Documented\\cutting_tests_processed\\2inch_stickout')

list_names = []
list_names.append('time_series_name_4p5inch.txt')
list_names.append('time_series_name_2inch.txt')

WFs = []
WFs.append('db10')
WFs.append('db10')

Levels = []
Levels.append(4)
Levels.append(4)

IWPs = []
IWPs.append(10)
IWPs.append(3)

label_names = []
label_names.append('4p5_inch_Labels_2Class.npy')
label_names.append('2_inch_Labels_2Class.npy')

samp_fs = []
samp_fs.append(10000)
samp_fs.append(10000)
   
saving = False
param_tuning=False
feature_ranking = False
cv=5

output = WPT_Transfer_Learning(data_paths,list_names,WFs,Levels,IWPs,label_names,samp_fs,cv,param_tuning,feature_ranking,saving)



from EEMD_Feature_Extraction import EEMD_Feature_Extraction
 
#parameters
stickout_length='3p5'
EEMDecs = 'A'
p=1
Classifier = 'SVC'

results = EEMD_Feature_Extraction(stickout_length, EEMDecs, 
                                  p, Classifier) 




# EEMD transfer learning
from EEMD_Transfer_Learning import EEMD_Transfer_Learning
 
#parameters
stickout_length_training='4p5'
stickout_length_test='2'
p_train = 1
p_test = 2
Classifier = 'SVC'

results = EEMD_Transfer_Learning(stickout_length_training, 
                                     stickout_length_test, 
                                     p_train, p_test,
                                     Classifier) 


features =results[2]

RankedList = results[4]

import numpy as np
how_many_times_rank = np.zeros((7,7))
for i in range (0,6):
    for j in range(0,10):
        a = RankedList[j][i][0]
        a = int(a)
        how_many_times_rank[a,i]=how_many_times_rank[a,i]+1

import scipy.io as sio
sio.savemat('number_of_times_feature_ranks_3p5_EEMD_SVM.mat',mdict={'times_feature_rank':how_many_times_rank})



# EEMD transfer learning 2 case

from EEMD_Transfer_Learning_2case import EEMD_Transfer_Learning_2case
stickout_lengths = ['2','2p5','3p5','4p5']
IMF_Number=[2,2,1,1]
Classifier = 'RF'

results = EEMD_Transfer_Learning_2case(stickout_lengths,IMF_Number,Classifier)



