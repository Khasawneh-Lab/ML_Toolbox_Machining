# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:44:24 2020

@author: yesillim
"""
# WPT 2-class classification

import numpy as np 
from WPT_Feature_Extraction import WPT_Feature_Extraction
stickout_length='4p5'
WPT_Level = 3
Classifier = 'SVC'
plotting = False

results = WPT_Feature_Extraction(stickout_length, WPT_Level, 
                                  Classifier, plotting)

# WPT transfer learning
import numpy as np 
from WPT_Transfer_Learning import WPT_Transfer_Learning
stickout_length_training = '2'
stickout_length_test = '4p5'
WPT_Level=4
Classifier='SVC'
results = WPT_Transfer_Learning(stickout_length_training, 
                                     stickout_length_test,
                                    WPT_Level, Classifier)     


# WPT transfer learning two case
from WPT_Transfer_Learning_2case import WPT_Transfer_Learning_2case
#parameters
 
stickout_lengths = ['2','2p5','3p5','4p5']
WPT_Level=4
Classifier='SVC'
results = WPT_Transfer_Learning_2case(stickout_lengths, 
                                       WPT_Level, Classifier)     

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



