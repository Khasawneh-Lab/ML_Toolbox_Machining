# -*- coding: utf-8 -*-
"""
Similarity Matrix Computation Using Dynamic Time Warping
--------------------------------------------------------

This algorithms computes the pairwise distances between time series obtained from 
the turning experiment explained in :cite:`Yesilli2019`. Dynamic Time Warping (DTW) is used
as similarity measure and this study uses the DTW package in this `link <https://pypi.org/project/cdtw/>`_.
Since DTW is commutative, we only compute the upper diagonal and the elements on the diagonal for 
the similarity matrix. 
This operation can be performed both in serial and parallel. This algorithm is the serial version. 
For parallel version of the same algorithm, please check this page.(add hyperlink here).
"""

#%% import the libraries
import time
from fastdtw import fastdtw
import pandas as pd
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import squareform
from itertools import combinations
from cdtw import pydtw
from scipy.spatial.distance import squareform
import sys 
import os.path

#%% define the case you want to compute the similarity/distance matrix
case = '3p5'

#%% add folder paths 
folderTosave= os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    'Cutting_Test_Data_Documented',
                                    'Distance_Matrix_for_Split_Data',
                                    )
sys.path.insert(0,folderTosave)

folderToLoad = os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    'Cutting_Test_Data_Documented',
                                    'Cutting_Data_Set_Divided_into_Groups',
                                    )
sys.path.insert(0,folderToLoad)
#%% load the dataset

TCD_Divided = np.load(os.path.join(folderToLoad,case+'_inch_Turning_Cutting_Data_Divided_Time_Series.npy'),allow_pickle=True)
Labels = np.load(os.path.join(folderToLoad,'Labels_'+case+'_inch_Turning_Cutting_Data_Divided_Time_Series.npy'),allow_pickle=True)


#%% cDTW(pip)
means = []
for i in range(len(TCD_Divided)):
    means.append(np.mean(TCD_Divided[i]))
    
#Combinations
poss_comb = np.array(list(combinations(range(0,len(TCD_Divided)), 2)))

# #distances between time series (DTW - Dynamic Time Warping)
N_Combination=len(poss_comb)
distance = np.zeros(shape=(N_Combination,1))

for i in range(0, N_Combination):
    start = time.time()
    x=np.ravel(TCD_Divided[poss_comb[i,0]])
    y=np.ravel(TCD_Divided[poss_comb[i,1]])
    d = pydtw.dtw(x,y,pydtw.Settings(compute_path = True)) 
    distance[i]=d.get_dist()    
    percent=(100*(i+1))/N_Combination
    end = time.time()
    duration = end-start
    print('Progress:{}  time:{}'.format(percent,duration))
    print(distance[i])
        

distance_matrix = squareform(np.ravel(distance))
# name = case+'_inch_Splitted_Data_Distance_Matrix_cDTW(pip)'
# os.path.join(folderTosave,name)
# np.save(os.path.join(folderTosave,name),distance_matrix)


