# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:17:13 2019

@author: yesillim
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 11:19:34 2019

@author: yesillim
"""

import time
start = time.time()
import numpy as np
import pandas as pd
import scipy.io as sio
import os.path
import sys
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew,kurtosis
from sklearn.feature_selection import RFE
from sklearn.svm import SVR,SVC,LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

#%%------------------------ADDING PATHS----------------------------------------

# folderToLoad_training ---> time series paths for training sets
# folderToLoad_test -------> time series paths for test sets
# folder_training ---------> IMF paths for training sets
# folder_test -------------> IMF paths for test set

#--------------------------TRAINING SET----------------------------------------

#folderToLoad_training1 = os.path.join('D:'+os.path.sep,
#                                    'Data Archive',
#                                    'Cutting Tool Experiment Data',
#                                    'DownSampled_Data',
#                                    '2inch',
#                                    )
#sys.path.insert(0,folderToLoad_training1)
#os.path.join(folderToLoad_training1)
#
#folder_training1 = os.path.join('D:'+os.path.sep,
#                                    'Data Archive',
#                                    "Cutting Tool Experiment Data",
#                                    'eIMFs',
#                                    '2inch',
#                                    )
#training_case_1 = 2

#folderToLoad_training2 = os.path.join('D:'+os.path.sep,
#                                    'Data Archive',
#                                    'Cutting Tool Experiment Data',
#                                    'DownSampled_Data',
#                                    '2.5inch',
#                                    )
#sys.path.insert(0,folderToLoad_training2)
#os.path.join(folderToLoad_training2)

#folder_training2 = os.path.join('D:'+os.path.sep,
#                                    'Data Archive',
#                                    "Cutting Tool Experiment Data",
#                                    'eIMFs',
#                                    '2.5inch',)
#training_case_2 = 2.5


folderToLoad_training1 = os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    'Cutting Tool Experiment Data',
                                    'DownSampled_Data',
                                    '3.5inch',
                                    )
sys.path.insert(0,folderToLoad_training1)
os.path.join(folderToLoad_training1)


folder_training1 = os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    "Cutting Tool Experiment Data",
                                    'eIMFs',
                                    '3.5inch',)
training_case_1 = 3.5

folderToLoad_training2 = os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    'Cutting Tool Experiment Data',
                                    'DownSampled_Data',
                                    '4.5 inc',
                                    )
sys.path.insert(0,folderToLoad_training2)
os.path.join(folderToLoad_training2)
#
folder_training2 = os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    'Cutting Tool Experiment Data',
                                    'eIMFs',
                                    '4.5inch',)

training_case_2 = 4.5

#%%--------------------------TEST SET--------------------------------------------

folderToLoad_test3 = os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    'Cutting Tool Experiment Data',
                                    'DownSampled_Data',
                                    '2inch',
                                    )
sys.path.insert(0,folderToLoad_test3)
os.path.join(folderToLoad_test3)

folder_test3 = os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    'Cutting Tool Experiment Data',
                                    'eIMFs',
                                    '2inch',
                                    )
test_case1 = 2

folderToLoad_test4 = os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    'Cutting Tool Experiment Data',
                                    'DownSampled_Data',
                                    '2.5inch',
                                    )
sys.path.insert(0,folderToLoad_test4)
os.path.join(folderToLoad_test4)

folder_test4 = os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    "Cutting Tool Experiment Data",
                                    'eIMFs',
                                    '2.5inch',)
test_case2 = 2.5

#folderToLoad_test3 = os.path.join('D:'+os.path.sep,
#                                    'Data Archive',
#                                    'Cutting Tool Experiment Data',
#                                    'DownSampled_Data',
#                                    '3.5inch',
#                                    )
#sys.path.insert(0,folderToLoad_test3)
#os.path.join(folderToLoad_test3)
#
#folder_test3 = os.path.join('D:'+os.path.sep,
#                                    'Data Archive',
#                                    "Cutting Tool Experiment Data",
#                                    'eIMFs',
#                                    '3.5inch',)
#
#test_case1 = 3.5
#
#folderToLoad_test4 = os.path.join('D:'+os.path.sep,
#                                    'Data Archive',
#                                    'Cutting Tool Experiment Data',
#                                    'DownSampled_Data',
#                                    '4.5 inc',
#                                    )
#sys.path.insert(0,folderToLoad_test4)
#os.path.join(folderToLoad_test4)
##
#folder_test4 = os.path.join('D:'+os.path.sep,
#                                    'Data Archive',
#                                    "Cutting Tool Experiment Data",
#                                    'eIMFs',
#                                    '4.5inch',)
#
#test_case2 = 4.5

#%%--------------CASE LABELS AND TIME SERIES NAMES-----------------------------

#for 2 inch case:
namets1 = ["c_320_005","c_425_020","c_425_025","c_570_001","c_570_002","c_570_005","c_570_010","c_770_001","c_770_002_2","c_770_002","c_770_005","c_770_010","i_320_005","i_320_010","i_425_020","i_425_025","i_570_002","i_570_005","i_570_010","i_770_001","s_320_005","s_320_010","s_320_015","s_320_020_2","s_320_020","s_320_025","s_320_030","s_320_035","s_320_040","s_320_045","s_320_050_2","s_320_050","s_425_005","s_425_010","s_425_015","s_425_017","s_425_020","s_570_002","s_570_005"]
label_c1=np.full((20,1),1)
label_u1=np.full((19,1),0)
label1=np.concatenate((label_c1,label_u1))
#label_training1 = label1
label_test1 =label1

#for 2.5 inch case:
namets2 = ["c_570_014","c_570_015s","c_770_005","i_570_012","i_570_014","i_570_015s","i_770_005","s_570_003","s_570_005_2","s_570_005","s_570_008","s_570_010","s_570_015_2","s_570_015","s_770_002","s_770_005"]
label_c2=np.full((7,1),1)
label_u2=np.full((9,1),0)
label2=np.concatenate((label_c2,label_u2))
#label_training2 = label2
label_test2 =label2

#for 3.5 inch case:
namets3 = ["c_1030_002","c_770_015","i_770_010","i_770_010_2","i_770_015","s_570_015","s_570_025","s_570_025_2","s_570_030","s_770_005","s_770_008","s_770_010","s_770_010_2","s_770_015"];
label_c3=np.full((5,1),1)
label_u3=np.full((9,1),0)
label3=np.concatenate((label_c3,label_u3))
label_training1 = label3
#label_test1 =label3

#for 4.5 inch case:
namets4 = ["c_570_035","c_570_040","c_1030_010","c_1030_015","c_1030_016","i_1030_010","i_1030_012","i_1030_013","i_1030_014","s_570_005","s_570_010","s_570_015","s_570_025","s_570_035","s_570_040","s_770_010","s_770_015","s_770_020","s_1030_005","s_1030_007","s_1030_013","s_1030_014"];
label_c4=np.full((9,1),1)
label_u4=np.full((13,1),0)
label4=np.concatenate((label_c4,label_u4))
label_training2 = label4
#label_test2 =label4

#%% length of datasets

#2 inch 
numberofcase1 = len(namets1)
#numberofcase_training1 = numberofcase1  #choose corresponding one by commenting out the other one 
numberofcase_test1 = numberofcase1

#2.5 inch 
numberofcase2 = len(namets2) 
#numberofcase_training2 = numberofcase2  #choose corresponding one by commenting out the other one 
numberofcase_test2 = numberofcase2

#3.5 inch
numberofcase3 = len(namets3) 
numberofcase_training1 = numberofcase3  #choose corresponding one by commenting out the other one 
#numberofcase_test1 = numberofcase3

#4.5 inch
numberofcase4 = len(namets4) 
numberofcase_training2 = numberofcase4  #choose corresponding one by commenting out the other one 
#numberofcase_test2 = numberofcase4
#%%-----------------------------LOAD TIMESERIES--------------------------------

#in this part of the code, by using name of time series we extract the rotatinal 
#speed and depth of cut information and store them in arrays for each case.

n_c_training = numberofcase1+numberofcase2
n_c_test = numberofcase3+numberofcase4

#training sets
labelrpm_training1 = np.zeros((numberofcase3,1))
labelrpm_training2 = np.zeros((numberofcase4,1))

labeldoc_training1 = np.zeros((numberofcase3,1))
labeldoc_training2 = np.zeros((numberofcase4,1))

#2 inch
for i in range (0,numberofcase_training1):
    name1 =  'chatter_1training%d' %(i+1)
    nameofdata1 = os.path.join(folderToLoad_training1,'%s_downsampled.mat' %(namets3[i]))
    if namets3[i][5]!='_':
        labelrpm_training1[i] = float(namets3[i][2:6])
        labeldoc_training1[i] = float(namets3[i][7:10])/1000
    else:
        labeldoc_training1[i] = float(namets3[i][2:5])
        labeldoc_training1[i] = float(namets3[i][6:9])/1000
    exec("%s = sio.loadmat(nameofdata1)" % (name1))
    exec('%s = %s["tsDS"]' %(name1,name1))
    
#2.5 inch
for i in range (0,numberofcase_training2):
    name1 =  'chatter_2training%d' %(i+1)
    nameofdata1 = os.path.join(folderToLoad_training2,'%s_downsampled.mat' %(namets4[i]))
    if namets4[i][5]!='_':
        labelrpm_training2[i] = float(namets4[i][2:6])
        labeldoc_training2[i] = float(namets4[i][7:10])/1000
    else:
        labelrpm_training2[i] = float(namets4[i][2:5])
        labeldoc_training2[i] = float(namets4[i][6:9])/1000 
    exec("%s = sio.loadmat(nameofdata1)" % (name1))
    exec('%s = %s["tsDS"]' %(name1,name1)) 
    
#3.5 inch
#for i in range (0,numberofcase_training):
#    name1 =  'chatter_training%d' %(i+1)
#    nameofdata1 = os.path.join(folderToLoad_training,'%s_downsampled.mat' %(namets3[i]))
#    if namets3[i][5]!='_':
#        labelrpm_training[i] = float(namets3[i][2:6])
#        labeldoc_training[i] = float(namets3[i][7:10])/1000
#    else:
#        labelrpm_training[i] = float(namets3[i][2:5])
#        labeldoc_training[i] = float(namets3[i][6:9])/1000      
#    exec("%s = sio.loadmat(nameofdata1)" % (name1))
#    exec('%s = %s["tsDS"]' %(name1,name1))    
    
#4.5 inch 
#for i in range (0,numberofcase_training):
#    name1 =  'chatter_training%d' %(i+1)
#    nameofdata1 = os.path.join(folderToLoad_training,'%s_downsampled.mat' %(namets4[i]))
#    if namets4[i][5]!='_':
#        labelrpm_training[i] = float(namets4[i][2:6])
#        labeldoc_training[i] = float(namets4[i][7:10])/1000
#    else:
#        labelrpm_training[i] = float(namets4[i][2:5])
#        labeldoc_training[i] = float(namets4[i][6:9])/1000   
#    exec("%s = sio.loadmat(nameofdata1)" % (name1))
#    exec('%s = %s["tsDS"]' %(name1,name1))



#test set
labelrpm_test1 = np.zeros((numberofcase1,1))
labelrpm_test2 = np.zeros((numberofcase2,1))

labeldoc_test1 = np.zeros((numberofcase1,1))
labeldoc_test2 = np.zeros((numberofcase2,1))

#2inch    

#for i in range (0,numberofcase_test1):
#    name2 =  'chatter_test%d' %(i+1)
#    nameofdata2 = os.path.join(folderToLoad_test,'%s_downsampled.mat' %(namets1[i]))
#    if namets1[i][5]!='_':
#        labelrpm_test[i] = float(namets1[i][2:6])
#        labeldoc_test[i] = float(namets1[i][7:10])/1000
#    else:
#        labelrpm_test[i] = float(namets1[i][2:5])
#        labeldoc_test[i] = float(namets1[i][6:9])/1000
#    exec("%s = sio.loadmat(nameofdata2)" % (name2))
#    exec('%s = %s["tsDS"]' %(name2,name2))

#2.5 inch
#for i in range (0,numberofcase_test):
#    name2 =  'chatter_test%d' %(i+1)
#    nameofdata2 = os.path.join(folderToLoad_test,'%s_downsampled.mat' %(namets2[i]))
#    if namets2[i][5]!='_':
#        labelrpm_test[i] = float(namets2[i][2:6])
#        labeldoc_test[i] = float(namets2[i][7:10])/1000
#    else:
#        labelrpm_test[i] = float(namets2[i][2:5])
#        labeldoc_test[i] = float(namets2[i][6:9])/1000    
#    exec("%s = sio.loadmat(nameofdata2)" % (name2))
#    exec('%s = %s["tsDS"]' %(name2,name2))
    
#3.5 inch
for i in range (0,numberofcase_test1):
    name2 =  'chatter_1test%d' %(i+1)
    nameofdata2 = os.path.join(folderToLoad_test3,'%s_downsampled.mat' %(namets1[i]))
    if namets1[i][5]!='_':
        labelrpm_test1[i] = float(namets1[i][2:6])
        labeldoc_test1[i] = float(namets1[i][7:10])/1000
    else:
        labelrpm_test1[i] = float(namets1[i][2:5])
        labeldoc_test1[i] = float(namets1[i][6:9])/1000
    exec("%s = sio.loadmat(nameofdata2)" % (name2))
    exec('%s = %s["tsDS"]' %(name2,name2))
   
#4.5 inch 
for i in range (0,numberofcase_test2):
    name2 =  'chatter_2test%d' %(i+1)
    nameofdata2 = os.path.join(folderToLoad_test4,'%s_downsampled.mat' %(namets2[i]))
    if namets2[i][5]!='_':
        labelrpm_test2[i] = float(namets2[i][2:6])
        labeldoc_test2[i] = float(namets2[i][7:10])/1000
    else:
        labelrpm_test2[i] = float(namets2[i][2:5])
        labeldoc_test2[i] = float(namets2[i][6:9])/1000 
    exec("%s = sio.loadmat(nameofdata2)" % (name2))
    exec('%s = %s["tsDS"]' %(name2,name2))

#%% splitting time series into small pieces in training set %TRAINING SET 1
chatter_data = []  #list which will store each time series for different cutting conditions
case_label = []    #list which will store labels for each value in time series

labels = np.concatenate((labelrpm_training1,labeldoc_training1,label_training1),axis=1)
for i in range (0, numberofcase_training1):
    name1 =  'chatter_1training%d' %(i+1)
    exec('chatter_data.append(%s)' %(name1))
    exec('L=len(%s)' %(name1))
    labelrpm=np.full((L,1),labels[i,0])
    labeldoc=np.full((L,1),labels[i,1])
    label=np.full((L,1),labels[i,2])
    labels_=np.concatenate((labelrpm,labeldoc,label),axis=1)
    case_label.append(labels_)

TCD_training1 = chatter_data


#length of each case. Use it to find the approximate number of division
length=np.zeros((numberofcase_training1,1))
for i in range(0,numberofcase_training1):
    length[i]=len(TCD_training1[i])
#------------------------------------------------------------------------------

caseLabels_training1 = np.zeros((1,3))  #intialize the matrix for labels

inc = 0                       # increment for total number of cases obtained after dividing

approximate_number_of_cases = int((sum(length))/1000)                     #approximate number of cases with respect to sum of lengths of actual cases

TCD_Divided_training1=np.ndarray(shape=(approximate_number_of_cases),dtype=object)  #create object array to store new time series obtained after split

for i in range(0,numberofcase_training1):                          
    data=TCD_training1[i]
    
    if len(data)>1000:
        division_number = int(len(data)/1000)            #number determines number of division for each time series    
        split=np.array_split(data,division_number)     #split data into different matrices not equal in size
        n=len(split)                                   #number of time series obtained after split
        Label=case_label[i][0]
        Label=np.reshape(Label,(1,3))
        
        for j in range(0,n):
            TCD_Divided_training1[inc]=np.array(split[j])                       #adding new time series into n dimensional array
            caseLabels_training1=np.append(caseLabels_training1,Label,axis=0)    #appending labels of each new time series into an array
            inc=inc+1
caseLabels_training1 = caseLabels_training1[1:]                                    #delete the first row of matrix to get rid of zeros 
TCD_Divided_training1 = TCD_Divided_training1[0:inc]                               #since this matrix dimension is defined with respect to approximate value at the begining, unused portion is ignored 

case = np.zeros((inc,1))
for i in range(0,inc):
    case[i]=i

caseLabels_training1=np.concatenate((caseLabels_training1,case),axis=1)         #add number to each time series
label_training1 =caseLabels_training1[:,2]

infoEMF_training1=np.ndarray(shape=(len(TCD_Divided_training1)),dtype=object)   #define an n dimensinal array to store IMFs 

#%% splitting time series into small pieces in training set %TRAINING SET 2
chatter_data = []  #list which will store each time series for different cutting conditions
case_label = []    #list which will store labels for each value in time series

labels = np.concatenate((labelrpm_training2,labeldoc_training2,label_training2),axis=1)
for i in range (0, numberofcase_training2):
    name1 =  'chatter_2training%d' %(i+1)
    exec('chatter_data.append(%s)' %(name1))
    exec('L=len(%s)' %(name1))
    labelrpm=np.full((L,1),labels[i,0])
    labeldoc=np.full((L,1),labels[i,1])
    label=np.full((L,1),labels[i,2])
    labels_=np.concatenate((labelrpm,labeldoc,label),axis=1)
    case_label.append(labels_)

TCD_training2 = chatter_data


#length of each case. Use it to find the approximate number of division
length=np.zeros((numberofcase_training2,1))
for i in range(0,numberofcase_training2):
    length[i]=len(TCD_training2[i])
#------------------------------------------------------------------------------

caseLabels_training2 = np.zeros((1,3))  #intialize the matrix for labels

inc = 0                       # increment for total number of cases obtained after dividing

approximate_number_of_cases = int((sum(length))/1000)                     #approximate number of cases with respect to sum of lengths of actual cases

TCD_Divided_training2=np.ndarray(shape=(approximate_number_of_cases),dtype=object)  #create object array to store new time series obtained after split

for i in range(0,numberofcase_training2):                          
    data=TCD_training2[i]
    
    if len(data)>1000:
        division_number = int(len(data)/1000)            #number determines number of division for each time series    
        split=np.array_split(data,division_number)     #split data into different matrices not equal in size
        n=len(split)                                   #number of time series obtained after split
        Label=case_label[i][0]
        Label=np.reshape(Label,(1,3))
        
        for j in range(0,n):
            TCD_Divided_training2[inc]=np.array(split[j])                       #adding new time series into n dimensional array
            caseLabels_training2=np.append(caseLabels_training2,Label,axis=0)    #appending labels of each new time series into an array
            inc=inc+1
caseLabels_training2 = caseLabels_training2[1:]                                    #delete the first row of matrix to get rid of zeros 
TCD_Divided_training2 = TCD_Divided_training2[0:inc]                               #since this matrix dimension is defined with respect to approximate value at the begining, unused portion is ignored 

case = np.zeros((inc,1))
for i in range(0,inc):
    case[i]=i

caseLabels_training2=np.concatenate((caseLabels_training2,case),axis=1)         #add number to each time series
label_training2 =caseLabels_training2[:,2]

infoEMF_training2=np.ndarray(shape=(len(TCD_Divided_training2)),dtype=object)   #define an n dimensinal array to store IMFs 



#%% splitting time series into small pieces in test set % TEST SET 1

chatter_data_test = []  #list which will store each time series for different cutting conditions
case_label_test = []    #list which will store labels for each value in time series

labels = np.concatenate((labelrpm_test1,labeldoc_test1,label_test1),axis=1)
for i in range (0, numberofcase_test1):
    name1 =  'chatter_1test%d' %(i+1)
    exec('chatter_data_test.append(%s)' %(name1))
    exec('L=len(%s)' %(name1))
    labelrpm=np.full((L,1),labels[i,0])
    labeldoc=np.full((L,1),labels[i,1])
    label=np.full((L,1),labels[i,2])
    labels_=np.concatenate((labelrpm,labeldoc,label),axis=1)
    case_label_test.append(labels_)

TCD_test1 = chatter_data_test


#length of each case. Use it to find the approximate number of division
length=np.zeros((numberofcase_test1,1))
for i in range(0,numberofcase_test1):
    length[i]=len(TCD_test1[i])
#------------------------------------------------------------------------------------------------------------------------------------------

caseLabels_test = np.zeros((1,3))  #intialize the matrix for labels

inc = 0                       # increment for total number of cases obtained after dividing

approximate_number_of_cases = int((sum(length))/1000)                     #approximate number of cases with respect to sum of lengths of actual cases

TCD_Divided_test1=np.ndarray(shape=(approximate_number_of_cases),dtype=object)  #create object array to store new time series obtained after split

for i in range(0,numberofcase_test1):                          
    data=TCD_test1[i]
    
    if len(data)>1000:
        division_number=int(len(data)/1000)            #number determines number of division for each time series    
        split=np.array_split(data,division_number)     #split data into different matrices not equal in size
        n=len(split)                                   #number of time series obtained after split
        Label=case_label_test[i][0]
        Label=np.reshape(Label,(1,3))

        for j in range(0,n):
            TCD_Divided_test1[inc]=np.array(split[j])              #adding new time series into n dimensional array
            caseLabels_test=np.append(caseLabels_test,Label,axis=0)    #appending labels of each new time series into an array
            inc=inc+1
caseLabels_test=caseLabels_test[1:]                                    #delete the first row of matrix to get rid of zeros 
TCD_Divided_test1=TCD_Divided_test1[0:inc]                               #since this matrix dimension is defined with respect to approximate value at the begining, unused portion is ignored 

case = np.zeros((inc,1))
for i in range(0,inc):
    case[i]=i

caseLabels_test1=np.concatenate((caseLabels_test,case),axis=1)         #add number to each time series
label_test1 =caseLabels_test[:,2]

infoEMF_test1=np.ndarray(shape=(len(TCD_Divided_test1)),dtype=object)   #define an n dimensinal array to store IMFs 

#%% splitting time series into small pieces in test set % TEST SET 2

chatter_data_test = []  #list which will store each time series for different cutting conditions
case_label_test = []    #list which will store labels for each value in time series

labels = np.concatenate((labelrpm_test2,labeldoc_test2,label_test2),axis=1)
for i in range (0, numberofcase_test2):
    name1 =  'chatter_2test%d' %(i+1)
    exec('chatter_data_test.append(%s)' %(name1))
    exec('L=len(%s)' %(name1))
    labelrpm=np.full((L,1),labels[i,0])
    labeldoc=np.full((L,1),labels[i,1])
    label=np.full((L,1),labels[i,2])
    labels_=np.concatenate((labelrpm,labeldoc,label),axis=1)
    case_label_test.append(labels_)

TCD_test2 = chatter_data_test


#length of each case. Use it to find the approximate number of division
length=np.zeros((numberofcase_test2,1))
for i in range(0,numberofcase_test2):
    length[i]=len(TCD_test2[i])
#------------------------------------------------------------------------------------------------------------------------------------------

caseLabels_test = np.zeros((1,3))  #intialize the matrix for labels

inc = 0                       # increment for total number of cases obtained after dividing

approximate_number_of_cases = int((sum(length))/1000)                     #approximate number of cases with respect to sum of lengths of actual cases

TCD_Divided_test2=np.ndarray(shape=(approximate_number_of_cases),dtype=object)  #create object array to store new time series obtained after split

for i in range(0,numberofcase_test2):                          
    data=TCD_test2[i]
    
    if len(data)>1000:
        division_number=int(len(data)/1000)            #number determines number of division for each time series    
        split=np.array_split(data,division_number)     #split data into different matrices not equal in size
        n=len(split)                                   #number of time series obtained after split
        Label=case_label_test[i][0]
        Label=np.reshape(Label,(1,3))

        for j in range(0,n):
            TCD_Divided_test2[inc]=np.array(split[j])              #adding new time series into n dimensional array
            caseLabels_test=np.append(caseLabels_test,Label,axis=0)    #appending labels of each new time series into an array
            inc=inc+1
caseLabels_test=caseLabels_test[1:]                                    #delete the first row of matrix to get rid of zeros 
TCD_Divided_test2=TCD_Divided_test2[0:inc]                               #since this matrix dimension is defined with respect to approximate value at the begining, unused portion is ignored 

case = np.zeros((inc,1))
for i in range(0,inc):
    case[i]=i

caseLabels_test2=np.concatenate((caseLabels_test,case),axis=1)         #add number to each time series
label_test2 =caseLabels_test[:,2]

infoEMF_test2=np.ndarray(shape=(len(TCD_Divided_test2)),dtype=object)   #define an n dimensinal array to store IMFs 



#%% load eIMFs if they are computed before

#training set


sys.path.insert(0,folder_training1)
lengthofcases = len(TCD_Divided_training1)
for i in range(0,lengthofcases):
    dataname = 'IMFs_%.1finch_Divided_Data_IMFs_Case%d' %(training_case_1,i+1) 
    infoEMF_training1[i] = sio.loadmat(os.path.join(folder_training1, dataname))
    infoEMF_training1[i] = infoEMF_training1[i]['eIMF']

sys.path.insert(0,folder_training2)
lengthofcases = len(TCD_Divided_training2)
for i in range(0,lengthofcases):
    dataname = 'IMFs_%.1finch_Divided_Data_IMFs_Case%d' %(training_case_2,i+1) 
    infoEMF_training2[i] = sio.loadmat(os.path.join(folder_training2, dataname))
    infoEMF_training2[i] = infoEMF_training2[i]['eIMF']


#test set

sys.path.insert(0,folder_test3)
lengthofcases = len(TCD_Divided_test1)
for i in range(0,lengthofcases):
    dataname = 'IMFs_%dinch_Divided_Data_IMFs_Case%d' %(test_case1,i+1) 
    infoEMF_test1[i] = sio.loadmat(os.path.join(folder_test3, dataname))
    infoEMF_test1[i] = infoEMF_test1[i]['eIMF']
    
sys.path.insert(0,folder_test4)
lengthofcases = len(TCD_Divided_test2)
for i in range(0,lengthofcases):
    dataname = 'IMFs_%.1finch_Divided_Data_IMFs_Case%d' %(test_case2,i+1) 
    infoEMF_test2[i] = sio.loadmat(os.path.join(folder_test4, dataname))
    infoEMF_test2[i] = infoEMF_test2[i]['eIMF']


#%%-------------------TRAINING SET FEATURES------------------------------------
#TRAINING SET 1
#define informative IMF number for TRAINING SET 1
p=2
features_training1=np.zeros((len(TCD_Divided_training1),7))

for i in range(0,len(TCD_Divided_training1)):
    eIMFs = infoEMF_training1[i]
    #feature_1
    nIMFs=len(eIMFs)
    A = np.power(eIMFs[p-1],2) 
    A_sum = sum(A)                                   #summing squares of whole elements of second IMF
    B_sum = 0               
    for k in range(nIMFs):
        B_sum = B_sum + sum(np.power(eIMFs[k],2))    #computing summing of squares of whole elements of IMFs
    features_training1[i,0]=A_sum/B_sum                        #energy ratio feature

for j in range(0,len(TCD_Divided_training1)):
   IntrinsicMFs=infoEMF_training1[j]
   #feature_2  Peak to peak value
   Maximum = max(IntrinsicMFs[p-1])
   Minimum = min(IntrinsicMFs[p-1])
   features_training1[j,1] = Maximum - Minimum 
   #feature_3 standard deviation
   features_training1[j,2] = np.std(IntrinsicMFs[p-1])
   #feature_4 root mean square (RMS)
   features_training1[j,3] = np.sqrt(np.mean(IntrinsicMFs[p-1]**2))   
   #feature_5 Crest factor
   features_training1[j,4] = Maximum/features_training1[j,3]
   #feature_6 Skewness
   features_training1[j,5] = skew(IntrinsicMFs[p-1])
   #feature_7 Kurtosis
   L= len(IntrinsicMFs[p-1])
   features_training1[j,6] = sum(np.power(IntrinsicMFs[p-1]-np.mean(IntrinsicMFs[p-1]),4)) / ((L-1)*np.power(features_training1[j,3],4))
   

#TRAINING SET 2
#define informative IMF number for TRAINING SET 2
p=2
features_training2=np.zeros((len(TCD_Divided_training2),7))

for i in range(0,len(TCD_Divided_training2)):
    eIMFs = infoEMF_training2[i]
    #feature_1
    nIMFs=len(eIMFs)
    A = np.power(eIMFs[p-1],2) 
    A_sum = sum(A)                                   #summing squares of whole elements of second IMF
    B_sum = 0               
    for k in range(nIMFs):
        B_sum = B_sum + sum(np.power(eIMFs[k],2))    #computing summing of squares of whole elements of IMFs
    features_training2[i,0]=A_sum/B_sum                        #energy ratio feature

for j in range(0,len(TCD_Divided_training2)):
   IntrinsicMFs=infoEMF_training2[j]
   #feature_2  Peak to peak value
   Maximum = max(IntrinsicMFs[p-1])
   Minimum = min(IntrinsicMFs[p-1])
   features_training2[j,1] = Maximum - Minimum 
   #feature_3 standard deviation
   features_training2[j,2] = np.std(IntrinsicMFs[p-1])
   #feature_4 root mean square (RMS)
   features_training2[j,3] = np.sqrt(np.mean(IntrinsicMFs[p-1]**2))   
   #feature_5 Crest factor
   features_training2[j,4] = Maximum/features_training2[j,3]
   #feature_6 Skewness
   features_training2[j,5] = skew(IntrinsicMFs[p-1])
   #feature_7 Kurtosis
   L= len(IntrinsicMFs[p-1])
   features_training2[j,6] = sum(np.power(IntrinsicMFs[p-1]-np.mean(IntrinsicMFs[p-1]),4)) / ((L-1)*np.power(features_training2[j,3],4))

#%%---------------------TEST SET FEATURES--------------------------------------

#TEST SET 1
features_test1=np.zeros((len(TCD_Divided_test1),7))
p=1
for i in range(0,len(TCD_Divided_test1)):
    eIMFs = infoEMF_test1[i]
    #feature_1
    nIMFs=len(eIMFs)
    A = np.power(eIMFs[p-1],2) 
    A_sum = sum(A)                                   #summing squares of whole elements of second IMF
    B_sum = 0               
    for k in range(nIMFs):
        B_sum = B_sum + sum(np.power(eIMFs[k],2))    #computing summing of squares of whole elements of IMFs
    features_test1[i,0]=A_sum/B_sum                        #energy ratio feature

for j in range(0,len(TCD_Divided_test1)):
   IntrinsicMFs=infoEMF_test1[j]
   #feature_2  Peak to peak value
   Maximum = max(IntrinsicMFs[p-1])
   Minimum = min(IntrinsicMFs[p-1])
   features_test1[j,1] = Maximum - Minimum 
   #feature_3 standard deviation
   features_test1[j,2] = np.std(IntrinsicMFs[p-1])
   #feature_4 root mean square (RMS)
   features_test1[j,3] = np.sqrt(np.mean(IntrinsicMFs[p-1]**2))   
   #feature_5 Crest factor
   features_test1[j,4] = Maximum/features_test1[j,3]
   #feature_6 Skewness
   features_test1[j,5] = skew(IntrinsicMFs[p-1])
   #feature_7 Kurtosis
   L= len(IntrinsicMFs[p-1])
   features_test1[j,6] = sum(np.power(IntrinsicMFs[p-1]-np.mean(IntrinsicMFs[p-1]),4)) / ((L-1)*np.power(features_test1[j,3],4))
   

#TEST SET 2
features_test2=np.zeros((len(TCD_Divided_test2),7))
p=1
for i in range(0,len(TCD_Divided_test2)):
    eIMFs = infoEMF_test2[i]
    #feature_1
    nIMFs=len(eIMFs)
    A = np.power(eIMFs[p-1],2) 
    A_sum = sum(A)                                   #summing squares of whole elements of second IMF
    B_sum = 0               
    for k in range(nIMFs):
        B_sum = B_sum + sum(np.power(eIMFs[k],2))    #computing summing of squares of whole elements of IMFs
    features_test2[i,0]=A_sum/B_sum                        #energy ratio feature

for j in range(0,len(TCD_Divided_test2)):
   IntrinsicMFs=infoEMF_test2[j]
   #feature_2  Peak to peak value
   Maximum = max(IntrinsicMFs[p-1])
   Minimum = min(IntrinsicMFs[p-1])
   features_test2[j,1] = Maximum - Minimum 
   #feature_3 standard deviation
   features_test2[j,2] = np.std(IntrinsicMFs[p-1])
   #feature_4 root mean square (RMS)
   features_test2[j,3] = np.sqrt(np.mean(IntrinsicMFs[p-1]**2))   
   #feature_5 Crest factor
   features_test2[j,4] = Maximum/features_test2[j,3]
   #feature_6 Skewness
   features_test2[j,5] = skew(IntrinsicMFs[p-1])
   #feature_7 Kurtosis
   L= len(IntrinsicMFs[p-1])
   features_test2[j,6] = sum(np.power(IntrinsicMFs[p-1]-np.mean(IntrinsicMFs[p-1]),4)) / ((L-1)*np.power(features_test2[j,3],4))

#%% CONCETANETE THE FEATURE MATRICES AND THE LABELS FOR TRAINING SET AND TEST SET
   
features_training = np.concatenate((features_training1,features_training2),axis=0)
features_test =  np.concatenate((features_test1,features_test2),axis=0)

label_training = np.concatenate((label_training1,label_training2),axis=0)
label_test = np.concatenate((label_test1,label_test2),axis=0)

#%%---------------CLASSIFICATION-----------------------------------------------

#creating train, test, accuracy, meanscore and deviation matrices
split_train_train,split_train_test = train_test_split(features_training, test_size=0.33)
split_test_train,split_test_test = train_test_split(features_test, test_size=0.33)

F_train_traincomb = np.zeros((len(split_train_train),7))
F_train_testcomb = np.zeros((len(split_train_test),7))
F_test_traincomb = np.zeros((len(split_test_train),7))
F_test_testcomb = np.zeros((len(split_test_test),7))

accuracy1 = np.zeros((7,10))
accuracy2 = np.zeros((7,10))

deviation1 = np.zeros((7,1))
deviation2 = np.zeros((7,1))

meanscore1 = np.zeros((7,1))
meanscore2 = np.zeros((7,1))

#repeat the procedure ten times 
Rank=[]
RankedList=[]
for o in range(0,10):
    
    #split into test and train set
    F_Training_Train,F_Training_Test,Label_Training_Train,Label_Training_Test= train_test_split(features_training,label_training, test_size=0.33)
    F_Test_Train,F_Test_Test,Label_Test_Train,Label_Test_Test= train_test_split(features_test,label_test, test_size=0.70)    
    
    #classification
#    clf = SVC(kernel='linear')
#    clf = LogisticRegression()
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
#    clf = GradientBoostingClassifier()
    
    #recursive feature elimination
    selector = RFE(clf, 1, step=1)
    Label_train=np.ravel(Label_Training_Train)
    Label_test =np.ravel(Label_Test_Test)
    selector = selector.fit(F_Training_Train, Label_train)
    rank = selector.ranking_
    Rank.append(rank)
    rank = np.asarray(rank)
    
    #create a list that contains index numbe of ranked features
    rankedlist = np.zeros((7,1))
    
    #finding index of the ranked features and creating new training and test sets with respect to this ranking
    for m in range (1,8):
        k=np.where(rank==m)
        rankedlist[m-1]=k[0][0]
        F_Training_Train[:,m-1] = F_Training_Train[:,int(rankedlist[m-1][0])]
        F_Test_Test[:,m-1] = F_Test_Test[:,int(rankedlist[m-1][0])] 
    RankedList.append(rankedlist)

    #trying various combinations of ranked features such as ([1],[1,2],[1,2,3]...)
    for p in range(0,7): 
        clf.fit(F_Training_Train[:,0:p+1],Label_train)
        score1=clf.score(F_Test_Test[:,0:p+1],Label_test)
        score2=clf.score(F_Training_Train[:,0:p+1],Label_train)
        accuracy1[p,o]=score1
        accuracy2[p,o]=score2
    print(o)
#computing mean score and deviation for each combination tried above        
for n in range(0,7):
    deviation1[n,0]=np.std(accuracy1[n,:])
    deviation2[n,0]=np.std(accuracy2[n,:])
    meanscore1[n,0]=np.mean(accuracy1[n,:])
    meanscore2[n,0]=np.mean(accuracy2[n,:])

results = np.concatenate((meanscore1,deviation1,meanscore2,deviation2),axis=1)
results = 100*results   
    
#total duration for algorithm  
end = time.time()
duration = end-start
print('Classification is completed in {} seconds.'.format(duration))

    
    
