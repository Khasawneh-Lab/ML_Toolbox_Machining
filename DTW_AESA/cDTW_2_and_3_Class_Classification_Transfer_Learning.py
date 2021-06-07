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
from itertools import permutations 

#%%cutting data set paramters


perm = list(permutations(['2', '2p5','3p5','4p5'],2))
perm2 = list(permutations(['2inch', '2p5inch','3p5inch','4p5inch'],2))

for pr in range(len(perm)):
    overh_dis_train = perm2[pr][0]
    case_train = perm[pr][0]
    
    overh_dis_test = perm2[pr][1]
    case_test = perm[pr][1]
    
    classification = '2Class'
    
    #define the random state numbers to make the comparison between different methods fair
    random_state_numbers = [35,21,18,40,13,17,34,29,36,11]
    #%% load data sets
    
    folderToLoad = os.path.join('D:'+os.path.sep,
                                        'Data Archive',
                                        'Cutting_Test_Data_Documented',
                                        'Distance_Matrix_for_Split_Data',
                                        )
    
    sys.path.insert(0,folderToLoad)
    
    name = case_train+'_'+case_test+'_Splitted_Data_Distance_Matrix_cDTW(pip)_'+classification[0]+'_Labels.npy'
    distance_matrix_test = np.load(os.path.join(folderToLoad,name))  
    
    name = case_train+'inch_Splitted_Data_Distance_Matrix_cDTW(pip)_'+classification[0]+'_Labels.npy'
    distance_matrix = np.load(os.path.join(folderToLoad,name))  
    
    
    
    folderToLoad = os.path.join('D:'+os.path.sep,
                                        'Data Archive',
                                        'Cutting_Test_Data_Documented',
                                        'Cutting_Data_Set_Divided_into_Groups',
                                        )
    sys.path.insert(0,folderToLoad)
    
    name = classification[0]+'_Labels_'+case_train+'_inch_Turning_Cutting_Data_Divided_Time_Series.npy'
    caseLabels_train = np.load(os.path.join(folderToLoad,name))  
    
    name = case_train+'_inch_Turning_Cutting_Data_Divided_Time_Series.npy'
    TCD_Divided_train = np.load(os.path.join(folderToLoad,name),allow_pickle=True)  
    
    name = classification[0]+'_Labels_'+case_test+'_inch_Turning_Cutting_Data_Divided_Time_Series.npy'
    caseLabels_test = np.load(os.path.join(folderToLoad,name))  
    
    name = case_test+'_inch_Turning_Cutting_Data_Divided_Time_Series.npy'
    TCD_Divided_test = np.load(os.path.join(folderToLoad,name),allow_pickle=True)   
    
    
    #%%
    
    #if we are training on milling data and testing on turning data
    if vice_versa:
        vv = TCD_Divided_train
        TCD_Divided_train = TCD_Divided_test
        TCD_Divided_test = vv
        
        vv2 = caseLabels_train
        caseLabels_train = caseLabels_test
        caseLabels_test = vv2
    
    
    #%% 2-Class
    
    if classification == '2Class':
    
        metrics_test =np.zeros((10,5))
        metrics_train =np.zeros((10,5))
        results = np.zeros((5,21))
        
        deviation_train = np.zeros((5,5))
        deviation_test = np.zeros((5,5))
        
        meanscore_train = np.zeros((5,5))
        meanscore_test = np.zeros((5,5))
        for k in range(1,6):
            start_time_classification = time.time()
            for T in range(0,10):
                #split time series into test set and training set
                TS_training_train,TS_training_test, Label_train_train, Label_train_test = train_test_split(TCD_Divided_train, caseLabels_train,random_state=random_state_numbers[T], test_size=0.33)
                TS_test_train,TS_test_test, Label_test_train, Label_test_test = train_test_split(TCD_Divided_test, caseLabels_test,random_state=random_state_numbers[T], test_size=0.67)
                
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
                tp=0
                fp=0
                tn=0
                fn=0
            
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
                            if test_class == 1:
                                tp= tp+1
                            else:
                                tn= tn+1
                        else:
                            false = false + 1
                            if test_class == 1:
                                fn= fn+1
                            else:
                                fp= fp+1
                                
                metrics_test[T,0] = true/(true+false) #accuracy
                
                #to avoid division by zero
                if (tp+fn) == 0:
                    recall=0
                else:
                    recall = tp/(tp+fn)
                #to avoid division by zero
                if (tp+fp) == 0:
                    precision = 0
                else:
                    precision = tp/(tp+fp)
                 #to avoid division by zero   
                if (precision+recall) == 0:
                    metrics_test[T,1] = 0
                else:
                    metrics_test[T,1] = 2*(precision*recall)/(precision+recall) # f1 score
                    
                metrics_test[T,2] = recall
                
                #to avoid division by zero
                if (tn+fp) == 0:
                    metrics_test[T,3] = 0
                else:
                    metrics_test[T,3] = tn/(tn+fp)   
                    
                metrics_test[T,4] = precision
                
         
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
                tp=0
                fp=0
                tn=0
                fn=0
                
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
                            if train_class == 1:
                                tp= tp+1
                            else:
                                tn= tn+1                        
                        else:
                            false = false + 1
                            if train_class == 1:
                                fn= fn+1
                            else:
                                fp= fp+1  
                                
                metrics_train[T,0] = true/(true+false) #accuracy
                
                #to avoid division by zero
                if (tp+fn) == 0:
                    recall=0
                else:
                    recall = tp/(tp+fn)
                #to avoid division by zero
                if (tp+fp) == 0:
                    precision = 0
                else:
                    precision = tp/(tp+fp)
                 #to avoid division by zero   
                if (precision+recall) == 0:
                    metrics_train[T,1] = 0
                else:
                    metrics_train[T,1] = 2*(precision*recall)/(precision+recall) # f1 score
                    
                metrics_train[T,2] = recall
                
                #to avoid division by zero
                if (tn+fp) == 0:
                    metrics_train[T,3] = 0
                else:
                    metrics_train[T,3] = tp/(tp+fp)   
                    
                metrics_train[T,4] = precision
                
                print('Iteration number {} is completed.'.format(T+1))
                     
            end_time_classification = time.time()
            classfication_time = end_time_classification - start_time_classification
            
            for v in range(5):
                deviation_test[k-1,v] = np.std(metrics_test[:,v])
                deviation_train[k-1,v] = np.std(metrics_train[:,v])
                meanscore_test[k-1,v] = np.mean(metrics_test[:,v])
                meanscore_train[k-1,v] = np.mean(metrics_train[:,v])
            
            print('k={}'.format(k))
            print('Score(Test Set): {} Deviation(Test Set): {}'.format(meanscore_test[k-1,0],deviation_test[k-1,0]))
            print('Score(Training Set): {} Deviation(Training Set): {}'.format(meanscore_train[k-1,0],deviation_train[k-1,0]))
            print('Classification Time: {}'.format(classfication_time))
            results[k-1,20]=classfication_time
            
        results[:,0:20] = np.concatenate((meanscore_test,deviation_test,meanscore_train,deviation_train),axis=1) 
        np.save('D:\Research Stuff\Transfer Learning on TDA Featurization\DTW_Similarity_Measure\Results\\'+case_train+'_'+case_test+'_TF_Learning_Results_cDTW(pip)_'+classification[0]+'_Labels.npy',results)
            
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
        # results[:,0:4] =results[:,0:4]*100