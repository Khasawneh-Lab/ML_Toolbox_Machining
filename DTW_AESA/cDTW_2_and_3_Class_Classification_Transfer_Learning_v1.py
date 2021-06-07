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

overh_dis_train = '2inch'
case_train = '2'

overh_dis_test = '4p5inch'
case_test = '4p5'

classification = '2Class'

#%% load data sets

folderToLoad = os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    'Cutting_Test_Data_Documented',
                                    'Distance_Matrix_for_Split_Data',
                                    )

sys.path.insert(0,folderToLoad)

name = case_train+'_'+case_test+'_Splitted_Data_Distance_Matrix_cDTW(pip)_3_Labels.npy'
distance_matrix_test = np.load(os.path.join(folderToLoad,name))  

name = case_train+'inch_Splitted_Data_Distance_Matrix_cDTW(pip)_3_Labels.npy'
distance_matrix = np.load(os.path.join(folderToLoad,name))  



folderToLoad = os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    'Cutting_Test_Data_Documented',
                                    'Cutting_Data_Set_Divided_into_Groups',
                                    )

sys.path.insert(0,folderToLoad)

name = '3_Labels_'+case_train+'_inch_Turning_Cutting_Data_Divided_Time_Series_Sorted_wrt_Labels.npy'
caseLabels_train = np.load(os.path.join(folderToLoad,name))  

name = case_train+'_inch_Turning_Cutting_Data_Divided_Time_Series_3_Labels_Sorted_wrt_Labels.npy'
TCD_Divided_train = np.load(os.path.join(folderToLoad,name),allow_pickle=True)  

name = '3_Labels_'+case_test+'_inch_Turning_Cutting_Data_Divided_Time_Series_Sorted_wrt_Labels.npy'
caseLabels_test = np.load(os.path.join(folderToLoad,name))  

name = case_test+'_inch_Turning_Cutting_Data_Divided_Time_Series_3_Labels_Sorted_wrt_Labels.npy'
TCD_Divided_test = np.load(os.path.join(folderToLoad,name),allow_pickle=True)   





#%%

if classification == '2Class':
    for i in range(len(TCD_Divided_train)):
        if caseLabels_train[i]==2:
            caseLabels_train[i]=1

    for j in range(len(TCD_Divided_test)):
        if caseLabels_test[j]==2:
            caseLabels_test[j]=1



#%% 2-Class

if classification == '2Class':

    accuracy_test =np.zeros((20))
    accuracy_train =np.zeros((20))
    results = np.zeros((5,5))
    
    for k in range(1,6):
        start_time_classification = time.time()
        for T in range(0,20):
            #split time series into test set and training set
            TS_training_train,TS_training_test, Label_train_train, Label_train_test = train_test_split(TCD_Divided_train, caseLabels_train, test_size=0.33)
            TS_test_train,TS_test_test, Label_test_train, Label_test_test = train_test_split(TCD_Divided_test, caseLabels_test, test_size=0.67)
            
            TS_train_L = len(TS_training_train)
            TS_test_L = len(TS_test_test)
             
            Label_train = Label_train_train
            Label_test = Label_test_test
            
            #find training set indices
            TS_train_index = np.zeros((TS_train_L))
            for i in range(0,TS_train_L):
                train_sample = TS_training_train[i]
                for j in range(0,len(TCD_Divided_train)):
                    if np.array_equal(TCD_Divided_train[j],train_sample):
                        TS_train_index[i] = j 
            #find test set indices
            TS_test_index = np.zeros((TS_test_L))
            for i in range(0,TS_test_L):
                test_sample = TS_test_test[i]
                for j in range(0,len(TCD_Divided_test)):
                    if np.array_equal(TCD_Divided_test[j],test_sample):
                        TS_test_index[i] = j             
            
            #generate distance matrix between test set and training set
            b = np.zeros(shape=(TS_train_L,TS_test_L))
            for i in range (0,TS_train_L):
                train_index = int(TS_train_index[i])
                for j in range(0,TS_test_L):
                    test_index = int(TS_test_index[j])
                    b[i,j]=distance_matrix_test[train_index,test_index]
                    
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
            
            #distance matrix for training set
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
            print('Iteration number {} is completed.'.format(T+1))
         
            
        mean_score_test = np.mean(accuracy_test)
        deviation_test=np.std(accuracy_test) 
        mean_score_train = np.mean(accuracy_train)
        deviation_train = np.std(accuracy_train)
        
        end_time_classification = time.time()
        classfication_time = end_time_classification - start_time_classification
        
        results[k-1,0] = mean_score_test
        results[k-1,1] = deviation_test
        results[k-1,2] = mean_score_train
        results[k-1,3] = deviation_train
        results[k-1,4] = classfication_time
    
        

        print('Score(Test Set): {} Deviation(Test Set): {}'.format(mean_score_test,deviation_test))
        print('Score(Training Set): {} Deviation(Training Set): {}'.format(mean_score_train,deviation_train))
        print('Classification Time: {}'.format(classfication_time))
    results[:,0:4] =results[:,0:4]*100
        
# 3-Class Classification

elif classification == '3Class':
    
    
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
            TS_training_train,TS_training_test, Label_train_train, Label_train_test = train_test_split(TCD_Divided_train, caseLabels_train, test_size=0.33)
            TS_test_train,TS_test_test, Label_test_train, Label_test_test = train_test_split(TCD_Divided_test, caseLabels_test, test_size=0.67)
            
            TS_train_L = len(TS_training_train)
            TS_test_L = len(TS_test_test)
             
            Label_train = Label_train_train
            Label_test = Label_test_test
            
            #find training set indices
            TS_train_index = np.zeros((TS_train_L))
            for i in range(0,TS_train_L):
                train_sample = TS_training_train[i]
                for j in range(0,len(TCD_Divided_train)):
                    if np.array_equal(TCD_Divided_train[j],train_sample):
                        TS_train_index[i] = j 
            #find test set indices
            TS_test_index = np.zeros((TS_test_L))
            for i in range(0,TS_test_L):
                test_sample = TS_test_test[i]
                for j in range(0,len(TCD_Divided_test)):
                    if np.array_equal(TCD_Divided_test[j],test_sample):
                        TS_test_index[i] = j             
            
            #generate distance matrix between test set and training set
            b = np.zeros(shape=(TS_train_L,TS_test_L))
            for i in range (0,TS_train_L):
                train_index = int(TS_train_index[i])
                for j in range(0,TS_test_L):
                    test_index = int(TS_test_index[j])
                    b[i,j]=distance_matrix_test[train_index,test_index]
                    
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
                    numberoftwos = pred_neighbour.count(2)
                    
                    countss=[]
                    countss.append(numberofzeros)
                    countss.append(numberofones)
                    countss.append(numberoftwos)
                    countss = np.asarray(countss)
                    max_count = max(countss)
                    pred = np.where(countss==max_count)[0][0]
                    
            
                    test_class = Label_test[i]
                    if test_class == pred:
                        true = true+1
                    else:
                        false = false + 1
            accuracy_test[T] = true/(true+false)
            
            #distance matrix for training set
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
                    numberoftwos = pred_neighbour.count(2)
                    
                    countss=[]
                    countss.append(numberofzeros)
                    countss.append(numberofones)
                    countss.append(numberoftwos)
                    countss = np.asarray(countss)
                    max_count = max(countss)
                    pred = np.where(countss==max_count)[0][0]
                    
                    test_class = Label_train[i]
                    if test_class == pred:
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