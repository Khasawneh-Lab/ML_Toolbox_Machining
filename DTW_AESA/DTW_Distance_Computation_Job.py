# -*- coding: utf-8 -*-
"""
Created on Sun May 24 15:06:42 2020

@author: yesillim
"""
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

m = int(sys.argv[2])
case = '3p5' 

if case == '2p5':
    npj=100
    if m != 83:
        if m!=0:
            k1 = m*npj-1
            k2 = (m+1)*npj-1
        else:
            k1 = m*npj
            k2 = (m+1)*npj-1        
    else: 
        k1 = 8299
        k2 = 8385

if case == '3p5':
    npj=100
    if m != 22:
        if m!=0:
            k1 = m*npj-1
            k2 = (m+1)*npj-1
        else:
            k1 = m*npj
            k2 = (m+1)*npj-1        
    else: 
        k1 = 2199
        k2 = 2211
        

if case == '4p5':
    npj=100
    if m != 153:
        if m!=0:
            k1 = m*npj-1
            k2 = (m+1)*npj-1
        else:
            k1 = m*npj
            k2 = (m+1)*npj-1        
    else: 
        k1 = 15299
        k2 = 15400


        
        
#%% paths include data and the functions required
folderTosave= os.path.join('/mnt'+os.path.sep,
                                    'ufs18',
                                    'home-175',
                                    'yesillim',
                                    'Documents',
                                    'Research',
                                    'DTW_Distance_Computation',
                                    'Distances',
                                    )
sys.path.insert(0,folderTosave)


folderToLoad = os.path.join('/mnt'+os.path.sep,
                                    'ufs18',
                                    'home-175',
                                    'yesillim',
                                    'Documents',
                                    'Research',
                                    'Data_Files',
                                    'Turning_Cutting_Data_Splitted',
                                    )

sys.path.insert(0,folderToLoad)


TCD_Divided = np.load(case+'_inch_Turning_Cutting_Data_Divided_Time_Series.npy',allow_pickle=True)
    
#%% cDTW(pip)
#Combinations
poss_comb = np.array(list(combinations(range(0,len(TCD_Divided)), 2)))

N_Combination=len(poss_comb)
distance = np.zeros(shape=(k2-k1,2))
inc=0
for i in range(k1, k2):
    x=np.ravel(TCD_Divided[(poss_comb[i,0])])
    y=np.ravel(TCD_Divided[(poss_comb[i,1])])
    d = pydtw.dtw(x,y,pydtw.Settings(compute_path = True)) 
    distance[inc,1]=d.get_dist() 
    distance[inc,0]=i
    percent=(100*(i+1))/N_Combination
    inc= inc+1
        
np.save(folderTosave+os.path.sep+'Distance_DTW'+str(case)+'inch_cases_'+str(k1)+'_'+str(k2)+'.npy', distance)

