# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 16:11:59 2020

@author: yesillim
"""
#import libraries
import numpy as np
import scipy.io as sio
import os, sys
from itertools import combinations
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from sklearn.model_selection import train_test_split
import math
from cdtw import pydtw
import time
matplotlib.rcParams.update({'font.size': 14})
rc('font',**{'family':'serif','serif':['Schoolbook']})
rc('text', usetex=True)


#%% parameters

overh_dist = '3p5inch'

#%% add paths to distances computed with parallel computing

folderToLoad = os.path.join('D:'+os.path.sep,
                                    'Research Stuff',
                                    'Paralell_Computing',
                                    'Distance_Matrix_Computation',
                                    'cDTW(pip)_Distances',
                                    overh_dist
                                    )

sys.path.insert(0,folderToLoad)
#load distances

DM= np.zeros((1,2))
listt= []

if overh_dist=='2inch':
    for m in range(175):
        npj=1000 
        if m != 174:
            if m!=0:
                k1 = m*npj-1
                k2 = (m+1)*npj-1
            else:
                k1 = m*npj
                k2 = (m+1)*npj-1 
        else: 
            k1 = 173999
            k2 = 174936
            
        dist = np.load(os.path.join(folderToLoad,'Distance_DTW2inch_cases_'+str(k1)+'_'+str(k2)+'_3_Labels.npy'))
        listt.append(dist)
        DM = np.concatenate((DM,dist), axis=0)
    DM=DM[1:]
    #convert it to distance matrix 
    DMat = squareform(DM[:,1])
    distance_matrix = DMat    

if overh_dist=='2p5inch':
    for m in range(82):
        npj=100
        if m != 81:
            if m!=0:
                k1 = m*npj-1
                k2 = (m+1)*npj-1
            else:
                k1 = m*npj
                k2 = (m+1)*npj-1        
        else: 
            k1 = 8099
            k2 = 8128
        dist = np.load(os.path.join(folderToLoad,'Distance_DTW2p5inch_cases_'+str(k1)+'_'+str(k2)+'_3_Labels.npy'))
        listt.append(dist)
        DM = np.concatenate((DM,dist), axis=0)
    DM=DM[1:]
    
    #convert it to distance matrix 
    DMat = squareform(DM[:,1])
    distance_matrix = DMat

if overh_dist=='3p5inch':
    for m in range(22):
        npj=100
        if m != 21:
            if m!=0:
                k1 = m*npj-1
                k2 = (m+1)*npj-1
            else:
                k1 = m*npj
                k2 = (m+1)*npj-1        
        else: 
            k1 = 2099
            k2 = 2145
        dist = np.load(os.path.join(folderToLoad,'Distance_DTW3p5inch_cases_'+str(k1)+'_'+str(k2)+'_3_Labels.npy'))
        listt.append(dist)
        DM = np.concatenate((DM,dist), axis=0)
    DM=DM[1:]
    
#convert it to distance matrix 
DMat = squareform(DM[:,1])
distance_matrix = DMat
    
if overh_dist=='4p5inch':   
    for m in range(154):
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
        dist = np.load(os.path.join(folderToLoad,'Distance_DTW4p5inch_cases_'+str(k1)+'_'+str(k2)+'_3_Labels.npy'))
        listt.append(dist)
        DM = np.concatenate((DM,dist), axis=0)
    DM=DM[1:]
    
#convert it to distance matrix 
DMat = squareform(DM[:,1])
distance_matrix = DMat    
    
#%% save distance matrix 


folderTosave= os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    'Cutting_Test_Data_Documented',
                                    'Distance_Matrix_for_Split_Data',
                                    )
sys.path.insert(0,folderTosave)

name = overh_dist+'_Splitted_Data_Distance_Matrix_cDTW(pip)_3_Labels'
os.path.join(folderTosave,name)
np.save(os.path.join(folderTosave,name),distance_matrix)
