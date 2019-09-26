# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:30:04 2019

@author: yesillim
"""

'''
Date : 4/8/2019

Author : Melih Can Yesilli

Feature matrix for turning experiment data are computed and features are ranked with 
recursive feature elimination method. Combinations of features are used to classify
chatter / no chatter cases for the experiment
 

-Update : This codes uses transfer learning principles.
'''
import time
start2 = time.time()
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import skew,kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.svm import SVR,SVC,LinearSVC
from sklearn.model_selection import train_test_split
from itertools import combinations
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib
import os, sys
matplotlib.rcParams.update({'font.size': 14})
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

#Add paths---------------------------------------------------------------------
folderToLoad1 = os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    'Cutting Tool Experiment Data',
                                    'DownSampled_Data',
                                    '2inch',
                                    )
sys.path.insert(0,folderToLoad1)
#os.path.join(folderToLoad1)

folderToLoad2 = os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    'Cutting Tool Experiment Data',
                                    'DownSampled_Data',
                                    '2.5inch',
                                    )
sys.path.insert(0,folderToLoad2)
#os.path.join(folderToLoad2)

folderToLoad3 = os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    'Cutting Tool Experiment Data',
                                    'DownSampled_Data',
                                    '3.5inch',
                                    )
sys.path.insert(0,folderToLoad3)
#os.path.join(folderToLoad3)

folderToLoad4 = os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    'Cutting Tool Experiment Data',
                                    'DownSampled_Data',
                                    '4.5 inc',
                                    )
sys.path.insert(0,folderToLoad4)
#os.path.join(folderToLoad4)
#------------------------------------------------------------------------------
#for 2 inch case:
namets1 = ["c_320_005","c_425_020","c_425_025","c_570_001","c_570_002","c_570_005","c_570_010","c_770_001","c_770_002_2","c_770_002","c_770_005","c_770_010","i_320_005","i_320_010","i_425_020","i_425_025","i_570_002","i_570_005","i_570_010","i_770_001","s_320_005","s_320_010","s_320_015","s_320_020_2","s_320_020","s_320_025","s_320_030","s_320_035","s_320_040","s_320_045","s_320_050_2","s_320_050","s_425_005","s_425_010","s_425_015","s_425_017","s_425_020","s_570_002","s_570_005"]
label_c1=np.full((20,1),1)
label_u1=np.full((19,1),0)
label1=np.concatenate((label_c1,label_u1))

#for 2.5 inch case:
namets2 = ["c_570_014","c_570_015s","c_770_005","i_570_012","i_570_014","i_570_015s","i_770_005","s_570_003","s_570_005_2","s_570_005","s_570_008","s_570_010","s_570_015_2","s_570_015","s_770_002","s_770_005"]
label_c2=np.full((7,1),1)
label_u2=np.full((9,1),0)
label2=np.concatenate((label_c2,label_u2))

#for 3.5 inch case:
namets3 = ["c_1030_002","c_770_015","i_770_010","i_770_010_2","i_770_015","s_570_015","s_570_025","s_570_025_2","s_570_030","s_770_005","s_770_008","s_770_010","s_770_010_2","s_770_015"];
label_c3=np.full((5,1),1)
label_u3=np.full((9,1),0)
label3=np.concatenate((label_c3,label_u3))

#for 4.5 inch case:
namets4 = ["c_570_035","c_570_040","c_1030_010","c_1030_015","c_1030_016","i_1030_010","i_1030_012","i_1030_013","i_1030_014","s_570_005","s_570_010","s_570_015","s_570_025","s_570_035","s_570_040","s_770_010","s_770_015","s_770_020","s_1030_005","s_1030_007","s_1030_013","s_1030_014"];
label_c4=np.full((9,1),1)
label_u4=np.full((13,1),0)
label4=np.concatenate((label_c4,label_u4))

#------------------------------------------------------------------------------

# length of datasets

#2 inch 
numberofcase1 = len(namets1)

#2.5 inch 
numberofcase2 = len(namets2) 

#3.5 inch
numberofcase3 = len(namets3) 

#4.5 inch
numberofcase4 = len(namets4) 

#%% load datasets

#training sets

#2 inch
#for i in range (0,numberofcase1):
#    name1 =  'chatter_1training_%d' %(i+1)
#    nameofdata1 = os.path.join(folderToLoad1,'WPT_Recon_%s.mat' %(namets1[i]))
#    exec("%s = sio.loadmat(nameofdata1)" % (name1))
#    exec('%s = %s["recon"]' %(name1,name1))
  
#2.5 inch
#for i in range (0,numberofcase2):
#    name1 =  'chatter_2training_%d' %(i+1)
#    nameofdata1 = os.path.join(folderToLoad2,'WPT_Recon_%s.mat' %(namets2[i]))
#    exec("%s = sio.loadmat(nameofdata1)" % (name1))
#    exec('%s = %s["recon"]' %(name1,name1)) 
    
#3.5 inch
for i in range (0,numberofcase3):
    name1 =  'chatter_3training_%d' %(i+1)
    nameofdata1 = os.path.join(folderToLoad3,'WPT_Recon_Level1_%s.mat' %(namets3[i]))
    exec("%s = sio.loadmat(nameofdata1)" % (name1))
    exec('%s = %s["recon"]' %(name1,name1))    
    
#4.5 inch 
for i in range (0,numberofcase4):
    name1 =  'chatter_4training_%d' %(i+1)
    nameofdata1 = os.path.join(folderToLoad4,'WPT_Recon_Level1_%s.mat' %(namets4[i]))
    exec("%s = sio.loadmat(nameofdata1)" % (name1))
    exec('%s = %s["recon"]' %(name1,name1))



#%%test set

#2inch    
for i in range (0,numberofcase1):
    name2 =  'chatter_1test_%d' %(i+1)
    nameofdata2 = os.path.join(folderToLoad1,'WPT_Level1_Recon_%s.mat' %(namets1[i]))
    exec("%s = sio.loadmat(nameofdata2)" % (name2))
    exec('%s = %s["recon"]' %(name2,name2))

#2.5 inch
for i in range (0,numberofcase2):
    name2 =  'chatter_2test_%d' %(i+1)
    nameofdata2 = os.path.join(folderToLoad2,'WPT_Recon_Level1_%s.mat' %(namets2[i]))
    exec("%s = sio.loadmat(nameofdata2)" % (name2))
    exec('%s = %s["recon"]' %(name2,name2))
    
#3.5 inch
#for i in range (0,numberofcase3):
#    name2 =  'chatter_3test_%d' %(i+1)
#    nameofdata2 = os.path.join(folderToLoad3,'WPT_Recon_%s.mat' %(namets3[i]))
#    exec("%s = sio.loadmat(nameofdata2)" % (name2))
#    exec('%s = %s["recon"]' %(name2,name2))
   
#4.5 inch 
#for i in range (0,numberofcase4):
#    name2 =  'chatter_4test_%d' %(i+1)
#    nameofdata2 = os.path.join(folderToLoad4,'WPT_Recon_%s.mat' %(namets4[i]))
#    exec("%s = sio.loadmat(nameofdata2)" % (name2))
#    exec('%s = %s["recon"]' %(name2,name2))   





#%%compute features for training set 

featuremat_training_1 = np.zeros((numberofcase3,10))
featuremat_training_2 = np.zeros((numberofcase4,10))
for i in range (0,numberofcase3):
    name =  'chatter_3training_%d' %(i+1)
    exec('ts1=%s' %name)
    featuremat_training_1[i,0] = np.average(ts1)
    featuremat_training_1[i,1] = np.std(ts1)
    featuremat_training_1[i,2] = np.sqrt(np.mean(ts1**2))   
    featuremat_training_1[i,3] = max(abs(ts1))
    featuremat_training_1[i,4] = skew(ts1)
    L=len(ts1)
    featuremat_training_1[i,5] = sum(np.power(ts1-featuremat_training_1[i,0],4)) / ((L-1)*np.power(featuremat_training_1[i,1],4))
    featuremat_training_1[i,6] = featuremat_training_1[i,3]/featuremat_training_1[i,2]
    featuremat_training_1[i,7] = featuremat_training_1[i,3]/np.power((np.average(np.sqrt(abs(ts1)))),2)
    featuremat_training_1[i,8] = featuremat_training_1[i,2]/(np.average((abs(ts1))))
    featuremat_training_1[i,9] = featuremat_training_1[i,3]/(np.average((abs(ts1))))
    
for i in range (0,numberofcase4):
    name =  'chatter_4training_%d' %(i+1)
    exec('ts1=%s' %name)
    featuremat_training_2[i,0] = np.average(ts1)
    featuremat_training_2[i,1] = np.std(ts1)
    featuremat_training_2[i,2] = np.sqrt(np.mean(ts1**2))   
    featuremat_training_2[i,3] = max(abs(ts1))
    featuremat_training_2[i,4] = skew(ts1)
    L=len(ts1)
    featuremat_training_2[i,5] = sum(np.power(ts1-featuremat_training_2[i,0],4)) / ((L-1)*np.power(featuremat_training_2[i,1],4))
    featuremat_training_2[i,6] = featuremat_training_2[i,3]/featuremat_training_2[i,2]
    featuremat_training_2[i,7] = featuremat_training_2[i,3]/np.power((np.average(np.sqrt(abs(ts1)))),2)
    featuremat_training_2[i,8] = featuremat_training_2[i,2]/(np.average((abs(ts1))))
    featuremat_training_2[i,9] = featuremat_training_2[i,3]/(np.average((abs(ts1))))

#compute features for test set
    
featuremat_test_1 = np.zeros((numberofcase1,10))
featuremat_test_2 = np.zeros((numberofcase2,10))

for i in range (0,numberofcase1):
    name =  'chatter_1test_%d' %(i+1)
    exec('ts2=%s' %name)
    featuremat_test_1[i,0] = np.average(ts2)
    featuremat_test_1[i,1] = np.std(ts2)
    featuremat_test_1[i,2] = np.sqrt(np.mean(ts2**2))   
    featuremat_test_1[i,3] = max(abs(ts2))
    featuremat_test_1[i,4] = skew(ts2)
    L=len(ts2)
    featuremat_test_1[i,5] = sum(np.power(ts2-featuremat_test_1[i,0],4)) / ((L-1)*np.power(featuremat_test_1[i,1],4))
    featuremat_test_1[i,6] = featuremat_test_1[i,3]/featuremat_test_1[i,2]
    featuremat_test_1[i,7] = featuremat_test_1[i,3]/np.power((np.average(np.sqrt(abs(ts2)))),2)
    featuremat_test_1[i,8] = featuremat_test_1[i,2]/(np.average((abs(ts2))))
    featuremat_test_1[i,9] = featuremat_test_1[i,3]/(np.average((abs(ts2))))    

for i in range (0,numberofcase2):
    name =  'chatter_2test_%d' %(i+1)
    exec('ts2=%s' %name)
    featuremat_test_2[i,0] = np.average(ts2)
    featuremat_test_2[i,1] = np.std(ts2)
    featuremat_test_2[i,2] = np.sqrt(np.mean(ts2**2))   
    featuremat_test_2[i,3] = max(abs(ts2))
    featuremat_test_2[i,4] = skew(ts2)
    L=len(ts2)
    featuremat_test_2[i,5] = sum(np.power(ts2-featuremat_test_2[i,0],4)) / ((L-1)*np.power(featuremat_test_2[i,1],4))
    featuremat_test_2[i,6] = featuremat_test_2[i,3]/featuremat_test_2[i,2]
    featuremat_test_2[i,7] = featuremat_test_2[i,3]/np.power((np.average(np.sqrt(abs(ts2)))),2)
    featuremat_test_2[i,8] = featuremat_test_2[i,2]/(np.average((abs(ts2))))
    featuremat_test_2[i,9] = featuremat_test_2[i,3]/(np.average((abs(ts2))))    


    
#%% load frequency domain feature computed in matlab for each case
    
#Features for training set
    
#2 inch case:
#freq_feature_data_name_training_set_1 = 'Freq_Features_2inch'
#freq_feature_data_name_training_set_1 = 'Freq_Features_2inch_WPT_Level1'

#2.5 inch case:
#freq_feature_data_name_training_set_2 = 'Freq_Features_2.5inch'
#freq_feature_data_name_training_set_2 = 'Freq_Features_2.5inch_WPT_Level1.mat'

#3.5 inch case:
#freq_feature_data_name_training_set_3 = 'Freq_Features_3.5inch'
freq_feature_data_name_training_set_3 = 'Freq_Features_3.5inch_Level1'

#4.5 inch case:
#freq_feature_data_name_training_set_4 = 'Freq_Features_4.5inch'
freq_feature_data_name_training_set_4 = 'Freq_Features_4.5inch_Level1'


#Features for test set
    
#2 inch case:
#freq_feature_data_name_test_set_1 = 'Freq_Features_2inch'
freq_feature_data_name_test_set_1 = 'Freq_Features_2inch_WPT_Level1'

#2.5 inch case:
#freq_feature_data_name_test_set_2 = 'Freq_Features_2.5inch'
freq_feature_data_name_test_set_2 = 'Freq_Features_2.5inch_WPT_Level1'

#3.5 inch case:
#freq_feature_data_name_test_set_3 = 'Freq_Features_3.5inch'
#freq_feature_data_name_test_set_3 = 'Freq_Features_3.5inch_Level1'

#4.5 inch case:
#freq_feature_data_name_test_set_4 = 'Freq_Features_4.5inch'
#freq_feature_data_name_test_set_4 = 'Freq_Features_4.5inch_Level1'   


freq_features_training_set_1 = sio.loadmat(os.path.join(folderToLoad3,freq_feature_data_name_training_set_3))
freq_features_training_set_1 = freq_features_training_set_1['Freq_Features']

freq_features_training_set_2 = sio.loadmat(os.path.join(folderToLoad4,freq_feature_data_name_training_set_4))
freq_features_training_set_2 = freq_features_training_set_2['Freq_Features']


freq_features_test_set_3 = sio.loadmat(os.path.join(folderToLoad1,freq_feature_data_name_test_set_1))
freq_features_test_set_3 = freq_features_test_set_3['Freq_Features']

freq_features_test_set_4 = sio.loadmat(os.path.join(folderToLoad2,freq_feature_data_name_test_set_2))
freq_features_test_set_4 = freq_features_test_set_4['Freq_Features']

#concatanate the frequency and time domain features 
featuremat_train1 = np.concatenate((featuremat_training_1, freq_features_training_set_1),axis = 1)
featuremat_train2 = np.concatenate((featuremat_training_2, freq_features_training_set_2),axis = 1)
featuremat_training  =  np.concatenate((featuremat_train1, featuremat_train2),axis = 0)

featuremat_test1 = np.concatenate((featuremat_test_1, freq_features_test_set_3),axis = 1)
featuremat_test2 = np.concatenate((featuremat_test_2, freq_features_test_set_4),axis = 1)
featuremat_test  = np.concatenate((featuremat_test1, featuremat_test2),axis = 0)

#%% labels of training set and test sets
label_train = np.concatenate((label3,label4),axis=0)
label_test = np.concatenate((label1,label2),axis=0)

#%%
#creating train, test, accuracy, meanscore and deviation matrices

F_traincomb = np.zeros((numberofcase1,14))
F_testcomb = np.zeros((numberofcase4,14))

accuracy1 = np.zeros((14,10))
accuracy2 = np.zeros((14,10))
deviation1 = np.zeros((14,1))
deviation2 = np.zeros((14,1))
meanscore1 = np.zeros((14,1))
meanscore2 = np.zeros((14,1))
duration1 = np.zeros((14,10))
meanduration = np.zeros((14,1))

#repeat the procedure ten times 
Rank=[]
RankedList=[]
for o in range(0,10):
    
    #split into test and train set
    F_Training_Train,F_Training_Test,Label_Training_Train,Label_Training_Test= train_test_split(featuremat_training,label_train, test_size=0.33)
    F_Test_Train,F_Test_Test,Label_Test_Train,Label_Test_Test= train_test_split(featuremat_test,label_test, test_size=0.70)
    
    #classification
#    clf = LinearSVC()
#    clf = LogisticRegression()
#    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf = GradientBoostingClassifier()


    #recursive feature elimination
    selector = RFE(clf, 1, step=1)
    Label_train=np.ravel(Label_Training_Train)
    Label_test =np.ravel(Label_Test_Test)
    selector = selector.fit(F_Training_Train, Label_train)
    rank = selector.ranking_
    Rank.append(rank)
    rank = np.asarray(rank)
    
    #create a list that contains index number of ranked features
    rankedlist = np.zeros((14,1))
    
    #classification
    print(o)
    #finding index of the ranked features and creating new training and test sets with respect to this ranking
    for m in range (1,15):
        k=np.where(rank==m)
        rankedlist[m-1]=k[0][0]
        F_Training_Train[:,m-1] = F_Training_Train[:,int(rankedlist[m-1][0])]
        F_Test_Test[:,m-1] = F_Test_Test[:,int(rankedlist[m-1][0])] 
    RankedList.append(rankedlist)
    print(o)
    #trying various combinations of ranked features such as ([1],[1,2],[1,2,3]...)
    for p in range(0,14): 
        start1 = time.time()
        clf.fit(F_Training_Train[:,0:p+1],Label_train)
        score1=clf.score(F_Test_Test[:,0:p+1],Label_test)
        score2=clf.score(F_Training_Train[:,0:p+1],Label_train)
        accuracy1[p,o]=score1
        accuracy2[p,o]=score2
        end1=time.time()
        duration1[p,o] = end1 - start1
    print(o)

#computing mean score and deviation for each combination tried above        
for n in range(0,14):
    deviation1[n,0]=np.std(accuracy1[n,:])
    deviation2[n,0]=np.std(accuracy2[n,:])
    meanscore1[n,0]=np.mean(accuracy1[n,:])
    meanscore2[n,0]=np.mean(accuracy2[n,:])
    meanduration[n,0]=np.mean(duration1[n,:])
    

results = np.concatenate((meanscore1,deviation1,meanscore2,deviation2),axis=1)
results = 100*results    
#total duration for algorithm  
end2 = time.time()
duration2 = end2-start2
print('Total elapsed time: {}'.format(duration2))
        

# This part of the code includes the ranked features for each iteration and keep them in arrays

#how_many_times_rank = np.zeros((14,14))
#for i in range (0,14):
#    for j in range(0,10):
#        a = RankedList[j][i][0]
#        a = int(a)
#        how_many_times_rank[a,i]=how_many_times_rank[a,i]+1
#
#sio.savemat('number_of_times_feature_ranks_4.5inch_WPT_Level4.mat',mdict={'times_feature_rank':how_many_times_rank})
