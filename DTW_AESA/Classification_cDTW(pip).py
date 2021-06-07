# -*- coding: utf-8 -*-
"""
Created on Fri May 29 16:35:45 2020

@author: yesillim

This codes takes the computed distance in HPCC and generate the distance matrix.
Then, it applies kNN classification to distance matrix to obtain the classification results.

"""

#import libraries
import numpy as np
import sys, os
from itertools import combinations
from scipy.spatial.distance import squareform
import scipy.io as sio
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time

#%%cutting data set paramters
overh_dist= '2p5inch'
case = '2p5'


#%% load data sets

folderToLoad = os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    'Cutting_Test_Data_Documented',
                                    'Distance_Matrix_for_Split_Data',
                                    )

sys.path.insert(0,folderToLoad)

name = overh_dist+'_Splitted_Data_Distance_Matrix_cDTW(pip).npy'
distance_matrix = np.load(os.path.join(folderToLoad,name))  
n_o_d = distance_matrix.shape[0]

folderToLoad = os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    'Cutting_Test_Data_Documented',
                                    'Cutting_Data_Set_Divided_into_Groups',
                                    )

sys.path.insert(0,folderToLoad)

name = 'Labels_'+case+'_inch_Turning_Cutting_Data_Divided_Time_Series.npy'
caseLabels = np.load(os.path.join(folderToLoad,name))  

name = case+'_inch_Turning_Cutting_Data_Divided_Time_Series.npy'
TCD_Divided = np.load(os.path.join(folderToLoad,name),allow_pickle=True)  


#%% 



accuracy_test =np.zeros((10))
accuracy_train =np.zeros((10))
results = np.zeros((5,5))
#number of nearest neighbors

k = 1
for c in range(5):
    start_time_classification = time.time()
    print(k)
    for T in range(0,10):
        #split time series into test set and training set
        TS_train,TS_test, Label_train, Label_test = train_test_split(TCD_Divided, caseLabels, test_size=0.33)
        
        TS_train_L = len(TS_train)
        TS_test_L = len(TS_test)
         
        Label_train = Label_train
        Label_test = Label_test
        
        #find training set indices
        TS_train_index = np.zeros((TS_train_L))
        for i in range(0,TS_train_L):
            train_sample = TS_train[i]
            for j in range(0,len(TCD_Divided)):
                if np.array_equal(TCD_Divided[j],train_sample):
                    TS_train_index[i] = j 
        #find test set indices
        TS_test_index = np.zeros((TS_test_L))
        for i in range(0,TS_test_L):
            test_sample = TS_test[i]
            for j in range(0,len(TCD_Divided)):
                if np.array_equal(TCD_Divided[j],test_sample):
                    TS_test_index[i] = j             
        
        #generate distance matrix between test set and training set
        b = np.zeros(shape=(TS_train_L,TS_test_L))
        for i in range (0,TS_train_L):
            train_index = int(TS_train_index[i])
            for j in range(0,TS_test_L):
                test_index = int(TS_test_index[j])
                b[i,j]=distance_matrix[train_index,test_index]
                
        #k-NN classification
        
        true = 0
        false = 0
        
        #test set classification
        
        for i in range(0,TS_test_L):
            if k==1:
                distances = b[:,i]
                min_dist = min(distances)
                index = np.where(distances == min_dist)[0][0]
                pred = Label_train[index]
                test_class = Label_test[i]
                if test_class == pred:
                    true = true+1
                else:
                    false = false + 1
            if k>1:
                distances = b[:,i]
                Sort_dist = sorted(distances)
                del Sort_dist[k:]
                index = np.zeros((k))
                for m in range (0,k):
                    index[m]=np.where(distances == Sort_dist[m])[0]
                index = index.astype(int)
                pred_neighbour = Label_train[index]
                pred_neighbour = pred_neighbour.tolist()
                numberofzeros = pred_neighbour.count(0)
                numberofones = pred_neighbour.count(1)
                if numberofzeros > numberofones:
                    pred = 0
                else:
                    pred = 1
                test_class = Label_test[i]
                if test_class == pred:
                    true = true+1
                else:
                    false = false + 1
        accuracy_test[T] = true/(true+false)
        
        b2 = np.zeros(shape=(TS_train_L,TS_train_L))
        for i in range (0,TS_train_L):
            train_index_1 = int(TS_train_index[i])
            for j in range(0,TS_train_L):
                train_index_2 = int(TS_train_index[j])
                b2[i,j]=distance_matrix[train_index_1,train_index_2]
        
        #training set classification
        
        true = 0
        false = 0
        for i in range(0,TS_train_L):
            if k==1:
                distances = b2[:,i]
                Sort_dist = sorted(distances)
                min_dist = Sort_dist[1]
                index = np.where(distances == min_dist)[0][0]
                pred = Label_train[index]
                train_class = Label_train[i]
                if train_class == pred:
                    true = true+1
                else:
                    false = false + 1
            if k>1:
                distances = b2[:,i]
                Sort_dist = sorted(distances)
                del Sort_dist[k:]
                index = np.zeros((k))
                for m in range (0,k):
                    index[m]=np.where(distances == Sort_dist[m])[0]
                index = index.astype(int)
                pred_neighbour = Label_train[index]
                pred_neighbour = pred_neighbour.tolist()
                numberofzeros = pred_neighbour.count(0)
                numberofones = pred_neighbour.count(1)
                if numberofzeros > numberofones:
                    pred = 0
                else:
                    pred = 1
                train_class = Label_train[i]
                if train_class == pred:
                    true = true+1
                else:
                    false = false + 1
        accuracy_train[T] = true/(true+false)
        
      
    mean_score_test = np.mean(accuracy_test)
    deviation_test=np.std(accuracy_test) 
    mean_score_train = np.mean(accuracy_train)
    deviation_train = np.std(accuracy_train)
    end_time_classification = time.time()
    classfication_time = end_time_classification - start_time_classification
    
    results[c,0] = mean_score_test
    results[c,1] = deviation_test
    results[c,2] = mean_score_train
    results[c,3] = deviation_train
    results[c,4] = classfication_time
    
    k=k+1
    print('Score(Test Set): {} Deviation(Test Set): {}'.format(mean_score_test,deviation_test))
    print('Score(Training Set): {} Deviation(Training Set): {}'.format(mean_score_train,deviation_train))
    print('Classification Time: {}'.format(classfication_time))
results[:,0:4] =results[:,0:4]*100


#%% for each k-NN, algorithm uses the same training and test set
# accuracy_test =np.zeros((10,5))
# accuracy_train =np.zeros((10,5))
# results = np.zeros((5,5))
# #number of nearest neighbors

# for T in range(0,10):
#     #split time series into test set and training set
#     TS_train,TS_test, Label_train, Label_test = train_test_split(TCD_Divided, caseLabels, test_size=0.33)
#     k = 1
#     for c in range(5):
#         start_time_classification = time.time()
#         print(k)
       
#         TS_train_L = len(TS_train)
#         TS_test_L = len(TS_test)
         
#         Label_train = Label_train
#         Label_test = Label_test
        
#         #find training set indices
#         TS_train_index = np.zeros((TS_train_L))
#         for i in range(0,TS_train_L):
#             train_sample = TS_train[i]
#             for j in range(0,len(TCD_Divided)):
#                 if np.array_equal(TCD_Divided[j],train_sample):
#                     TS_train_index[i] = j 
#         #find test set indices
#         TS_test_index = np.zeros((TS_test_L))
#         for i in range(0,TS_test_L):
#             test_sample = TS_test[i]
#             for j in range(0,len(TCD_Divided)):
#                 if np.array_equal(TCD_Divided[j],test_sample):
#                     TS_test_index[i] = j             
        
#         #generate distance matrix between test set and training set
#         b = np.zeros(shape=(TS_train_L,TS_test_L))
#         for i in range (0,TS_train_L):
#             train_index = int(TS_train_index[i])
#             for j in range(0,TS_test_L):
#                 test_index = int(TS_test_index[j])
#                 b[i,j]=distance_matrix[train_index,test_index]
                
#         #k-NN classification
        
#         true = 0
#         false = 0
        
#         #test set classification
        
#         for i in range(0,TS_test_L):
#             if k==1:
#                 distances = b[:,i]
#                 min_dist = min(distances)
#                 index = np.where(distances == min_dist)[0][0]
#                 pred = Label_train[index]
#                 test_class = Label_test[i]
#                 if test_class == pred:
#                     true = true+1
#                 else:
#                     false = false + 1
#             if k>1:
#                 distances = b[:,i]
#                 Sort_dist = sorted(distances)
#                 del Sort_dist[k:]
#                 index = np.zeros((k))
#                 for m in range (0,k):
#                     index[m]=np.where(distances == Sort_dist[m])[0]
#                 index = index.astype(int)
#                 pred_neighbour = Label_train[index]
#                 pred_neighbour = pred_neighbour.tolist()
#                 numberofzeros = pred_neighbour.count(0)
#                 numberofones = pred_neighbour.count(1)
#                 if numberofzeros > numberofones:
#                     pred = 0
#                 else:
#                     pred = 1
#                 test_class = Label_test[i]
#                 if test_class == pred:
#                     true = true+1
#                 else:
#                     false = false + 1
#         accuracy_test[T,c] = true/(true+false)
        
#         b2 = np.zeros(shape=(TS_train_L,TS_train_L))
#         for i in range (0,TS_train_L):
#             train_index_1 = int(TS_train_index[i])
#             for j in range(0,TS_train_L):
#                 train_index_2 = int(TS_train_index[j])
#                 b2[i,j]=distance_matrix[train_index_1,train_index_2]
        
#         #training set classification
        
#         true = 0
#         false = 0
#         for i in range(0,TS_train_L):
#             if k==1:
#                 distances = b2[:,i]
#                 Sort_dist = sorted(distances)
#                 min_dist = Sort_dist[1]
#                 index = np.where(distances == min_dist)[0][0]
#                 pred = Label_train[index]
#                 train_class = Label_train[i]
#                 if train_class == pred:
#                     true = true+1
#                 else:
#                     false = false + 1
#             if k>1:
#                 distances = b2[:,i]
#                 Sort_dist = sorted(distances)
#                 del Sort_dist[k:]
#                 index = np.zeros((k))
#                 for m in range (0,k):
#                     index[m]=np.where(distances == Sort_dist[m])[0]
#                 index = index.astype(int)
#                 pred_neighbour = Label_train[index]
#                 pred_neighbour = pred_neighbour.tolist()
#                 numberofzeros = pred_neighbour.count(0)
#                 numberofones = pred_neighbour.count(1)
#                 if numberofzeros > numberofones:
#                     pred = 0
#                 else:
#                     pred = 1
#                 train_class = Label_train[i]
#                 if train_class == pred:
#                     true = true+1
#                 else:
#                     false = false + 1
#         accuracy_train[T,c] = true/(true+false)
#         k=k+1

# for i in range(5):
#     mean_score_test = np.mean(accuracy_test[:,i])
#     deviation_test=np.std(accuracy_test[:,i]) 
#     mean_score_train = np.mean(accuracy_train[:,i])
#     deviation_train = np.std(accuracy_train[:,i])

#     results[i,0] = mean_score_test
#     results[i,1] = deviation_test
#     results[i,2] = mean_score_train
#     results[i,3] = deviation_train

# results[:,0:4] =results[:,0:4]*100


