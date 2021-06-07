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

train_case = '4p5'
test_case = '2'
#%% add paths to distances computed with parallel computing

folderToLoad = os.path.join('D:'+os.path.sep,
                                    'Research Stuff',
                                    'Paralell_Computing',
                                    'Distance_Matrix_Computation',
                                    'cDTW(pip)_Distances',
                                    '2p5-4p5',
                                    )

sys.path.insert(0,folderToLoad)

#load distances

DM = np.zeros((176,593))
for i in range(176):
    name = 'Distance_DTW'+train_case+'_'+test_case+'inch_cases_'+str(i)+'train_sample.npy'
    path = os.path.join(folderToLoad,name)
    DM[i,:]=np.load(path,allow_pickle=True)[:,1]


#%% save distance matrix 


folderTosave= os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    'Cutting_Test_Data_Documented',
                                    'Distance_Matrix_for_Split_Data',
                                    )
sys.path.insert(0,folderTosave)

name = train_case+'_'+test_case+'_Splitted_Data_Distance_Matrix_cDTW(pip)_2_Labels'
os.path.join(folderTosave,name)
np.save(os.path.join(folderTosave,name),DM)
