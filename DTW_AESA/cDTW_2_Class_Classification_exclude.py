# -*- coding: utf-8 -*-
"""
Created on Fri May 29 16:35:45 2020

@author: yesillim

"""

#import libraries
import sys, os
from sklearn.model_selection import train_test_split
import time
from itertools import combinations
import scipy.io as sio
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
from matplotlib.ticker import FixedLocator, FixedFormatter
import pandas as pd
import matplotlib.pyplot as plt
from itertools import groupby
import numpy as np 
from matplotlib import rc
from matplotlib.colors import SymLogNorm, LogNorm
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
### for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


#%%cutting data set paramters
overh_dist= '2inch'
case = '2'
#%% load data sets

folderToLoad = os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    'Cutting_Test_Data_Documented',
                                    'Distance_Matrix_for_Split_Data',
                                    )

sys.path.insert(0,folderToLoad)

name = overh_dist+'_Splitted_Data_Distance_Matrix_cDTW(pip)_3_Labels.npy'
distance_matrix = np.load(os.path.join(folderToLoad,name))  
n_o_d = distance_matrix.shape[0]

folderToLoad = os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    'Cutting_Test_Data_Documented',
                                    'Cutting_Data_Set_Divided_into_Groups',
                                    )

sys.path.insert(0,folderToLoad)

name = '3_Labels_'+case+'_inch_Turning_Cutting_Data_Divided_Time_Series.npy'
caseLabels = np.load(os.path.join(folderToLoad,name))  

name = case+'_inch_Turning_Cutting_Data_Divided_Time_Series_3_Labels.npy'
TCD_Divided = np.load(os.path.join(folderToLoad,name),allow_pickle=True)  





#%%whole distance matrix plot 
caseLabels = np.reshape(caseLabels,(len(caseLabels),1))
labeled_distancematrix = np.concatenate((caseLabels,distance_matrix),axis=1)

#%% find the index of each labels in distance matrix
index_zeros=[]
index_ones = []
index_twos = []
for i in range(len(TCD_Divided)):
    if labeled_distancematrix[i,0]==0:
        index_zeros.append(i)
    elif labeled_distancematrix[i,0]==1: 
        index_ones.append(i)
    elif labeled_distancematrix[i,0]==2: 
        index_twos.append(i)


label_sorted_dist_matrix = np.zeros((len(index_zeros)+len(index_ones),len(index_zeros)+len(index_ones)))

labels = []

for i in range(len(index_zeros)):
    labels.append(caseLabels[index_zeros[i]])

for j in range(len(index_ones)):
    labels.append(caseLabels[index_ones[j]])

labels = np.asarray(labels)


TCD_Divided_sorted = np.zeros((len(index_zeros)+len(index_ones)),dtype=object)
inc=0
for k in range(len(TCD_Divided)):
    if caseLabels[k]==0 or caseLabels[k]==1:
        TCD_Divided_sorted[inc] = TCD_Divided[k]
        inc=inc+1

caseLabels = labels
TCD_Divided = TCD_Divided_sorted
#%% stable-stable distance_matrix
stable_stable = np.zeros((len(index_zeros),len(index_zeros)))
for i in range(len(index_zeros)):
    for j in range(len(index_zeros)):
        ind1 = index_zeros[i]
        ind2 = index_zeros[j]
        stable_stable[i,j] = distance_matrix[ind1,ind2]

#%% stable-intermediate distance_matrix

# stable_int = np.zeros((len(index_zeros),len(index_twos)))
# for i in range(len(index_zeros)):
#     for j in range(len(index_twos)):
#         ind1 = index_zeros[i]
#         ind2 = index_twos[j]
#         stable_int[i,j] = distance_matrix[ind1,ind2]

#%% stable-chatter

stable_chat = np.zeros((len(index_zeros),len(index_ones)))
for i in range(len(index_zeros)):
    for j in range(len(index_ones)):
        ind1 = index_zeros[i]
        ind2 = index_ones[j]
        stable_chat[i,j] = distance_matrix[ind1,ind2]
#%% int-stable
# int_stable = np.zeros((len(index_twos),len(index_zeros)))
# for i in range(len(index_twos)):
#     for j in range(len(index_zeros)):
#         ind1 = index_twos[i]
#         ind2 = index_zeros[j]
#         int_stable[i,j] = distance_matrix[ind1,ind2]

# #%% int-int
# int_int = np.zeros((len(index_twos),len(index_twos)))
# for i in range(len(index_twos)):
#     for j in range(len(index_twos)):
#         ind1 = index_twos[i]
#         ind2 = index_twos[j]
#         int_int[i,j] = distance_matrix[ind1,ind2]

# #%% int-chat
# int_chat = np.zeros((len(index_twos),len(index_ones)))
# for i in range(len(index_twos)):
#     for j in range(len(index_ones)):
#         ind1 = index_twos[i]
#         ind2 = index_ones[j]
#         int_chat[i,j] = distance_matrix[ind1,ind2]

#%% chat-stable
chat_stable = np.zeros((len(index_ones),len(index_zeros)))
for i in range(len(index_ones)):
    for j in range(len(index_zeros)):
        ind1 = index_ones[i]
        ind2 = index_zeros[j]
        chat_stable[i,j] = distance_matrix[ind1,ind2]

#%% chat-int
# chat_int = np.zeros((len(index_ones),len(index_twos)))
# for i in range(len(index_ones)):
#     for j in range(len(index_twos)):
#         ind1 = index_ones[i]
#         ind2 = index_twos[j]
#         chat_int[i,j] = distance_matrix[ind1,ind2]

#%% chat-chat
chat_chat = np.zeros((len(index_ones),len(index_ones)))
for i in range(len(index_ones)):
    for j in range(len(index_ones)):
        ind1 = index_ones[i]
        ind2 = index_ones[j]
        chat_chat[i,j] = distance_matrix[ind1,ind2]

#%% combine these matrices to form label sorted distance matrix

label_sorted_dist_matrix[0:len(index_zeros),0:len(index_zeros)] = stable_stable
# label_sorted_dist_matrix[0:len(index_zeros),len(index_zeros):(len(index_twos)+len(index_zeros))] = stable_int
label_sorted_dist_matrix[0:len(index_zeros),len(index_zeros):(len(index_zeros)+len(index_ones))] = stable_chat

# label_sorted_dist_matrix[len(index_zeros):(len(index_zeros)+len(index_twos)),0:len(index_zeros)] = int_stable
# label_sorted_dist_matrix[len(index_zeros):(len(index_zeros)+len(index_twos)),len(index_zeros):(len(index_twos)+len(index_zeros))] = int_int
# label_sorted_dist_matrix[len(index_zeros):(len(index_zeros)+len(index_twos)),(len(index_twos)+len(index_zeros)):(len(index_twos)+len(index_zeros)+len(index_ones))] = int_chat

label_sorted_dist_matrix[len(index_zeros):(len(index_zeros)+len(index_ones)),0:len(index_zeros)] = chat_stable
# label_sorted_dist_matrix[(len(index_zeros)+len(index_twos)):(len(index_twos)+len(index_zeros)+len(index_ones)),len(index_zeros):(len(index_twos)+len(index_zeros))] = chat_int
label_sorted_dist_matrix[len(index_zeros):(len(index_zeros)+len(index_ones)),len(index_zeros):(len(index_zeros)+len(index_ones))] = chat_chat

#%% check if the new sorted dist matrix is symmetric
check = []
for i in range(len(index_zeros)+len(index_ones)):
    for j in range(len(index_zeros)+len(index_ones)):
        dist1 = label_sorted_dist_matrix[i,j]
        dist2 = label_sorted_dist_matrix[j,i]
        
        if dist1==dist2:
            check.append(1)
        else:
            check.append(0)

# if this section prints True, this means that the sorted distance matrix is symmetric
check = np.asarray(check)
print(np.all(check==1))        

#%% 

import time

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


