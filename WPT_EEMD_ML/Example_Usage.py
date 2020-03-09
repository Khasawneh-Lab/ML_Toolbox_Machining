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
stickout_length='2p5'
EEMDecs = 'A'
p=2
Classifier = 'SVC'

results = EEMD_Feature_Extraction(stickout_length, EEMDecs, 
                                  p, Classifier) 

features =results[2]

# EEMD transfer learning
from EEMD_Transfer_Learning import EEMD_Transfer_Learning
 
#parameters
stickout_length_training='2'
stickout_length_test='4p5'
p_train = 2
p_test = 1
Classifier = 'GB'

results = EEMD_Transfer_Learning(stickout_length_training, 
                                     stickout_length_test, 
                                     p_train, p_test,
                                     Classifier) 

# EEMD transfer learning 2 case

from EEMD_Transfer_Learning_2case import EEMD_Transfer_Learning_2case
stickout_lengths = ['2','2p5','3p5','4p5']
IMF_Number=[2,2,1,1]
Classifier = 'RF'

results = EEMD_Transfer_Learning_2case(stickout_lengths,IMF_Number,Classifier)



