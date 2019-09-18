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

#folderToLoad_training = os.path.join('D:'+os.path.sep,
#                                    'Data Archive',
#                                    'Cutting Tool Experiment Data',
#                                    'DownSampled_Data',
#                                    '2inch',
#                                    )
#sys.path.insert(0,folderToLoad_training)
#os.path.join(folderToLoad_training)
#
#folder_training = os.path.join('D:'+os.path.sep,
#                                    'Data Archive',
#                                    "Cutting Tool Experiment Data",
#                                    'eIMFs',
#                                    '2inch',
#                                    )
#training_case = 2

#folderToLoad_training = os.path.join('D:'+os.path.sep,
#                                    'Data Archive',
#                                    'Cutting Tool Experiment Data',
#                                    'DownSampled Data',
#                                    '2.5inch',
#                                    )
#sys.path.insert(0,folderToLoad_training)
#os.path.join(folderToLoad_training)

#folder_training = os.path.join('D:'+os.path.sep,
#                                    'Data Archive',
#                                    "Cutting Tool Experiment Data",
#                                    'eIMFs',
#                                    '2.5inch',)
#training_case = 2.5


#folderToLoad_training = os.path.join('D:'+os.path.sep,
#                                    'Data Archive',
#                                    'Cutting Tool Experiment Data',
#                                    'DownSampled Data',
#                                    '3.5inch',
#                                    )
#sys.path.insert(0,folderToLoad_training)
#os.path.join(folderToLoad_training)


#folder_training = os.path.join('D:'+os.path.sep,
#                                    'Data Archive',
#                                    "Cutting Tool Experiment Data",
#                                    'eIMFs',
#                                    '3.5inch',)
#training_case = 3.5

folderToLoad_training = os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    'Cutting Tool Experiment Data',
                                    'DownSampled_Data',
                                    '4.5 inc',
                                    )
sys.path.insert(0,folderToLoad_training)
#os.path.join(folderToLoad_test)
#
folder_training = os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    'Cutting Tool Experiment Data',
                                    'eIMFs',
                                    '4.5inch',)

training_case = 4.5

#%%--------------------------TEST SET--------------------------------------------

folderToLoad_test = os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    'Cutting Tool Experiment Data',
                                    'DownSampled_Data',
                                    '2inch',
                                    )
sys.path.insert(0,folderToLoad_test)
#os.path.join(folderToLoad_test)
#
folder_test = os.path.join('D:'+os.path.sep,
                                    'Data Archive',
                                    'Cutting Tool Experiment Data',
                                    'eIMFs',
                                    '2inch',
                                    )
test_case = 2

#folderToLoad_test = os.path.join('D:'+os.path.sep,
#                                    'Data Archive',
#                                    'Cutting Tool Experiment Data',
#                                    'DownSampled Data',
#                                    '2.5inch',
#                                    )
#sys.path.insert(0,folderToLoad_test)
#os.path.join(folderToLoad_test)
#
#folder_test = os.path.join('D:'+os.path.sep,
#                                    'Data Archive',
#                                    "Cutting Tool Experiment Data",
#                                    'eIMFs',
#                                    '2.5inch',)
#test_case = 2.5

#folderToLoad_test = os.path.join('D:'+os.path.sep,
#                                    'Data Archive',
#                                    'Cutting Tool Experiment Data',
#                                    'DownSampled Data',
#                                    '3.5inch',
#                                    )
#sys.path.insert(0,folderToLoad_test)
#os.path.join(folderToLoad_test)

#folder_test = os.path.join('D:'+os.path.sep,
#                                    'Data Archive',
#                                    "Cutting Tool Experiment Data",
#                                    'eIMFs',
#                                    '3.5inch',)
#
#test_case = 3.5

#folderToLoad_test = os.path.join('D:'+os.path.sep,
#                                    'Data Archive',
#                                    'Cutting Tool Experiment Data',
#                                    'DownSampled_Data',
#                                    '4.5 inc',
#                                    )
#sys.path.insert(0,folderToLoad_test)
#os.path.join(folderToLoad_test)
##
#folder_test = os.path.join('D:'+os.path.sep,
#                                    'Data Archive',
#                                    "Cutting Tool Experiment Data",
#                                    'eIMFs',
#                                    '4.5inch',)
#
#test_case = 4.5

#%%--------------CASE LABELS AND TIME SERIES NAMES-----------------------------

#for 2 inch case:
namets1 = ["c_320_005","c_425_020","c_425_025","c_570_001","c_570_002","c_570_005","c_570_010","c_770_001","c_770_002_2","c_770_002","c_770_005","c_770_010","i_320_005","i_320_010","i_425_020","i_425_025","i_570_002","i_570_005","i_570_010","i_770_001","s_320_005","s_320_010","s_320_015","s_320_020_2","s_320_020","s_320_025","s_320_030","s_320_035","s_320_040","s_320_045","s_320_050_2","s_320_050","s_425_005","s_425_010","s_425_015","s_425_017","s_425_020","s_570_002","s_570_005"]
label_c1=np.full((20,1),1)
label_u1=np.full((19,1),0)
label1=np.concatenate((label_c1,label_u1))
#label_training = label1
label_test =label1

#for 2.5 inch case:
#namets2 = ["c_570_014","c_570_015s","c_770_005","i_570_012","i_570_014","i_570_015s","i_770_005","s_570_003","s_570_005_2","s_570_005","s_570_008","s_570_010","s_570_015_2","s_570_015","s_770_002","s_770_005"]
#label_c2=np.full((7,1),1)
#label_u2=np.full((9,1),0)
#label2=np.concatenate((label_c2,label_u2))
##label_training = label2
#label_test =label2

#for 3.5 inch case:
#namets3 = ["c_1030_002","c_770_015","i_770_010","i_770_010_2","i_770_015","s_570_015","s_570_025","s_570_025_2","s_570_030","s_770_005","s_770_008","s_770_010","s_770_010_2","s_770_015"];
#label_c3=np.full((5,1),1)
#label_u3=np.full((9,1),0)
#label3=np.concatenate((label_c3,label_u3))
#label_training = label3
#label_test =label3

#for 4.5 inch case:
namets4 = ["c_570_035","c_570_040","c_1030_010","c_1030_015","c_1030_016","i_1030_010","i_1030_012","i_1030_013","i_1030_014","s_570_005","s_570_010","s_570_015","s_570_025","s_570_035","s_570_040","s_770_010","s_770_015","s_770_020","s_1030_005","s_1030_007","s_1030_013","s_1030_014"];
label_c4=np.full((9,1),1)
label_u4=np.full((13,1),0)
label4=np.concatenate((label_c4,label_u4))
label_training = label4
#label_test =label4

#%% length of datasets

#2 inch 
numberofcase1 = len(namets1)
#numberofcase_training = numberofcase1  #choose corresponding one by commenting out the other one 
numberofcase_test = numberofcase1

#2.5 inch 
#numberofcase2 = len(namets2) 
#numberofcase_training = numberofcase2  #choose corresponding one by commenting out the other one 
#numberofcase_test = numberofcase2

#3.5 inch
#numberofcase3 = len(namets3) 
#numberofcase_training = numberofcase3  #choose corresponding one by commenting out the other one 
#numberofcase_test = numberofcase3

#4.5 inch
numberofcase4 = len(namets4) 
numberofcase_training = numberofcase4  #choose corresponding one by commenting out the other one 
#numberofcase_test = numberofcase4
#%%-----------------------------LOAD TIMESERIES--------------------------------

#in this part of the code, by using name of time series we extract the rotatinal 
#speed and depth of cut information and store them in arrays for each case.

#training sets
labelrpm_training = np.zeros((numberofcase_training,1))
labeldoc_training = np.zeros((numberofcase_training,1))
#2 inch
#for i in range (0,numberofcase_training):
#    name1 =  'chatter_training%d' %(i+1)
#    nameofdata1 = os.path.join(folderToLoad_training,'%s_downsampled.mat' %(namets1[i]))
#    if namets1[i][5]!='_':
#        labelrpm_training[i] = float(namets1[i][2:6])
#        labeldoc_training[i] = float(namets1[i][7:10])/1000
#    else:
#        labelrpm_training[i] = float(namets1[i][2:5])
#        labeldoc_training[i] = float(namets1[i][6:9])/1000
#    exec("%s = sio.loadmat(nameofdata1)" % (name1))
#    exec('%s = %s["tsDS"]' %(name1,name1))
    
#2.5 inch
#for i in range (0,numberofcase_training):
#    name1 =  'chatter_training%d' %(i+1)
#    nameofdata1 = os.path.join(folderToLoad_training,'%s_downsampled.mat' %(namets2[i]))
#    if namets2[i][5]!='_':
#        labelrpm_training[i] = float(namets2[i][2:6])
#        labeldoc_training[i] = float(namets2[i][7:10])/1000
#    else:
#        labelrpm_training[i] = float(namets2[i][2:5])
#        labeldoc_training[i] = float(namets2[i][6:9])/1000 
#    exec("%s = sio.loadmat(nameofdata1)" % (name1))
#    exec('%s = %s["tsDS"]' %(name1,name1)) 
    
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
for i in range (0,numberofcase_training):
    name1 =  'chatter_training%d' %(i+1)
    nameofdata1 = os.path.join(folderToLoad_training,'%s_downsampled.mat' %(namets4[i]))
    if namets4[i][5]!='_':
        labelrpm_training[i] = float(namets4[i][2:6])
        labeldoc_training[i] = float(namets4[i][7:10])/1000
    else:
        labelrpm_training[i] = float(namets4[i][2:5])
        labeldoc_training[i] = float(namets4[i][6:9])/1000   
    exec("%s = sio.loadmat(nameofdata1)" % (name1))
    exec('%s = %s["tsDS"]' %(name1,name1))



#test set
labelrpm_test = np.zeros((numberofcase_test,1))
labeldoc_test = np.zeros((numberofcase_test,1))

#2inch    

for i in range (0,numberofcase_test):
    name2 =  'chatter_test%d' %(i+1)
    nameofdata2 = os.path.join(folderToLoad_test,'%s_downsampled.mat' %(namets1[i]))
    if namets1[i][5]!='_':
        labelrpm_test[i] = float(namets1[i][2:6])
        labeldoc_test[i] = float(namets1[i][7:10])/1000
    else:
        labelrpm_test[i] = float(namets1[i][2:5])
        labeldoc_test[i] = float(namets1[i][6:9])/1000
    exec("%s = sio.loadmat(nameofdata2)" % (name2))
    exec('%s = %s["tsDS"]' %(name2,name2))

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
#for i in range (0,numberofcase_test):
#    name2 =  'chatter_test%d' %(i+1)
#    nameofdata2 = os.path.join(folderToLoad_test,'%s_downsampled.mat' %(namets3[i]))
#    if namets3[i][5]!='_':
#        labelrpm_test[i] = float(namets3[i][2:6])
#        labeldoc_test[i] = float(namets3[i][7:10])/1000
#    else:
#        labelrpm_test[i] = float(namets3[i][2:5])
#        labeldoc_test[i] = float(namets3[i][6:9])/1000
#    exec("%s = sio.loadmat(nameofdata2)" % (name2))
#    exec('%s = %s["tsDS"]' %(name2,name2))
   
#4.5 inch 
#for i in range (0,numberofcase_test):
#    name2 =  'chatter_test%d' %(i+1)
#    nameofdata2 = os.path.join(folderToLoad_test,'%s_downsampled.mat' %(namets4[i]))
#    if namets4[i][5]!='_':
#        labelrpm_test[i] = float(namets4[i][2:6])
#        labeldoc_test[i] = float(namets4[i][7:10])/1000
#    else:
#        labelrpm_test[i] = float(namets4[i][2:5])
#        labeldoc_test[i] = float(namets4[i][6:9])/1000 
#    exec("%s = sio.loadmat(nameofdata2)" % (name2))
#    exec('%s = %s["tsDS"]' %(name2,name2))

#%% splitting time series into small pieces in training set
chatter_data = []  #list which will store each time series for different cutting conditions
case_label = []    #list which will store labels for each value in time series

labels = np.concatenate((labelrpm_training,labeldoc_training,label_training),axis=1)
for i in range (0, numberofcase_training):
    name1 =  'chatter_training%d' %(i+1)
    exec('chatter_data.append(%s)' %(name1))
    exec('L=len(%s)' %(name1))
    labelrpm=np.full((L,1),labels[i,0])
    labeldoc=np.full((L,1),labels[i,1])
    label=np.full((L,1),labels[i,2])
    labels_=np.concatenate((labelrpm,labeldoc,label),axis=1)
    case_label.append(labels_)

TCD_training = chatter_data


#length of each case. Use it to find the approximate number of division
length=np.zeros((numberofcase_training,1))
for i in range(0,numberofcase_training):
    length[i]=len(TCD_training[i])
#------------------------------------------------------------------------------

caseLabels_training = np.zeros((1,3))  #intialize the matrix for labels

inc = 0                       # increment for total number of cases obtained after dividing

approximate_number_of_cases = int((sum(length))/1000)                     #approximate number of cases with respect to sum of lengths of actual cases

TCD_Divided_training=np.ndarray(shape=(approximate_number_of_cases),dtype=object)  #create object array to store new time series obtained after split

for i in range(0,numberofcase_training):                          
    data=TCD_training[i]
    
    if len(data)>1000:
        division_number = int(len(data)/1000)            #number determines number of division for each time series    
        split=np.array_split(data,division_number)     #split data into different matrices not equal in size
        n=len(split)                                   #number of time series obtained after split
        Label=case_label[i][0]
        Label=np.reshape(Label,(1,3))
        
        for j in range(0,n):
            TCD_Divided_training[inc]=np.array(split[j])                       #adding new time series into n dimensional array
            caseLabels_training=np.append(caseLabels_training,Label,axis=0)    #appending labels of each new time series into an array
            inc=inc+1
caseLabels_training = caseLabels_training[1:]                                    #delete the first row of matrix to get rid of zeros 
TCD_Divided_training = TCD_Divided_training[0:inc]                               #since this matrix dimension is defined with respect to approximate value at the begining, unused portion is ignored 

case = np.zeros((inc,1))
for i in range(0,inc):
    case[i]=i

caseLabels_training=np.concatenate((caseLabels_training,case),axis=1)         #add number to each time series
label_training =caseLabels_training[:,2]

infoEMF_training=np.ndarray(shape=(len(TCD_Divided_training)),dtype=object)   #define an n dimensinal array to store IMFs 

#%% splitting time series into small pieces in test set

chatter_data_test = []  #list which will store each time series for different cutting conditions
case_label_test = []    #list which will store labels for each value in time series

labels = np.concatenate((labelrpm_test,labeldoc_test,label_test),axis=1)
for i in range (0, numberofcase_test):
    name1 =  'chatter_test%d' %(i+1)
    exec('chatter_data_test.append(%s)' %(name1))
    exec('L=len(%s)' %(name1))
    labelrpm=np.full((L,1),labels[i,0])
    labeldoc=np.full((L,1),labels[i,1])
    label=np.full((L,1),labels[i,2])
    labels_=np.concatenate((labelrpm,labeldoc,label),axis=1)
    case_label_test.append(labels_)

TCD_test = chatter_data_test


#length of each case. Use it to find the approximate number of division
length=np.zeros((numberofcase_test,1))
for i in range(0,numberofcase_test):
    length[i]=len(TCD_test[i])
#------------------------------------------------------------------------------------------------------------------------------------------

caseLabels_test = np.zeros((1,3))  #intialize the matrix for labels

inc = 0                       # increment for total number of cases obtained after dividing

approximate_number_of_cases = int((sum(length))/1000)                     #approximate number of cases with respect to sum of lengths of actual cases

TCD_Divided_test=np.ndarray(shape=(approximate_number_of_cases),dtype=object)  #create object array to store new time series obtained after split

for i in range(0,numberofcase_test):                          
    data=TCD_test[i]
    
    if len(data)>1000:
        division_number=int(len(data)/1000)            #number determines number of division for each time series    
        split=np.array_split(data,division_number)     #split data into different matrices not equal in size
        n=len(split)                                   #number of time series obtained after split
        Label=case_label_test[i][0]
        Label=np.reshape(Label,(1,3))

        for j in range(0,n):
            TCD_Divided_test[inc]=np.array(split[j])              #adding new time series into n dimensional array
            caseLabels_test=np.append(caseLabels_test,Label,axis=0)    #appending labels of each new time series into an array
            inc=inc+1
caseLabels_test=caseLabels_test[1:]                                    #delete the first row of matrix to get rid of zeros 
TCD_Divided_test=TCD_Divided_test[0:inc]                               #since this matrix dimension is defined with respect to approximate value at the begining, unused portion is ignored 

case = np.zeros((inc,1))
for i in range(0,inc):
    case[i]=i

caseLabels_test=np.concatenate((caseLabels_test,case),axis=1)         #add number to each time series
label_test =caseLabels_test[:,2]

infoEMF_test=np.ndarray(shape=(len(TCD_Divided_test)),dtype=object)   #define an n dimensinal array to store IMFs 


#%% load eIMFs if they are computed before

#training set

#define informative IMF number for TRAINING SET
p=1

sys.path.insert(0,folder_training)
lengthofcases = len(TCD_Divided_training)
for i in range(0,lengthofcases):
    dataname = 'IMFs_%.1finch_Divided_Data_IMFs_Case%d' %(training_case,i+1) 
    infoEMF_training[i] = sio.loadmat(os.path.join(folder_training, dataname))
    infoEMF_training[i] = infoEMF_training[i]['eIMF']

#test set
sys.path.insert(0,folder_test)
lengthofcases = len(TCD_Divided_test)
for i in range(0,lengthofcases):
    dataname = 'IMFs_%dinch_Divided_Data_IMFs_Case%d' %(test_case,i+1) 
    infoEMF_test[i] = sio.loadmat(os.path.join(folder_test, dataname))
    infoEMF_test[i] = infoEMF_test[i]['eIMF']


#%%-------------------TRAINING SET FEATURES------------------------------------

features_training=np.zeros((len(TCD_Divided_training),7))

for i in range(0,len(TCD_Divided_training)):
    eIMFs = infoEMF_training[i]
    #feature_1
    nIMFs=len(eIMFs)
    A = np.power(eIMFs[p-1],2) 
    A_sum = sum(A)                                   #summing squares of whole elements of second IMF
    B_sum = 0               
    for k in range(nIMFs):
        B_sum = B_sum + sum(np.power(eIMFs[k],2))    #computing summing of squares of whole elements of IMFs
    features_training[i,0]=A_sum/B_sum                        #energy ratio feature

for j in range(0,len(TCD_Divided_training)):
   IntrinsicMFs=infoEMF_training[j]
   #feature_2  Peak to peak value
   Maximum = max(IntrinsicMFs[p-1])
   Minimum = min(IntrinsicMFs[p-1])
   features_training[j,1] = Maximum - Minimum 
   #feature_3 standard deviation
   features_training[j,2] = np.std(IntrinsicMFs[p-1])
   #feature_4 root mean square (RMS)
   features_training[j,3] = np.sqrt(np.mean(IntrinsicMFs[p-1]**2))   
   #feature_5 Crest factor
   features_training[j,4] = Maximum/features_training[j,3]
   #feature_6 Skewness
   features_training[j,5] = skew(IntrinsicMFs[p-1])
   #feature_7 Kurtosis
   L= len(IntrinsicMFs[p-1])
   features_training[j,6] = sum(np.power(IntrinsicMFs[p-1]-np.mean(IntrinsicMFs[p-1]),4)) / ((L-1)*np.power(features_training[j,3],4))

#%%---------------------TEST SET FEATURES--------------------------------------
features_test=np.zeros((len(TCD_Divided_test),7))
p=2
for i in range(0,len(TCD_Divided_test)):
    eIMFs = infoEMF_test[i]
    #feature_1
    nIMFs=len(eIMFs)
    A = np.power(eIMFs[p-1],2) 
    A_sum = sum(A)                                   #summing squares of whole elements of second IMF
    B_sum = 0               
    for k in range(nIMFs):
        B_sum = B_sum + sum(np.power(eIMFs[k],2))    #computing summing of squares of whole elements of IMFs
    features_test[i,0]=A_sum/B_sum                        #energy ratio feature

for j in range(0,len(TCD_Divided_test)):
   IntrinsicMFs=infoEMF_test[j]
   #feature_2  Peak to peak value
   Maximum = max(IntrinsicMFs[p-1])
   Minimum = min(IntrinsicMFs[p-1])
   features_test[j,1] = Maximum - Minimum 
   #feature_3 standard deviation
   features_test[j,2] = np.std(IntrinsicMFs[p-1])
   #feature_4 root mean square (RMS)
   features_test[j,3] = np.sqrt(np.mean(IntrinsicMFs[p-1]**2))   
   #feature_5 Crest factor
   features_test[j,4] = Maximum/features_test[j,3]
   #feature_6 Skewness
   features_test[j,5] = skew(IntrinsicMFs[p-1])
   #feature_7 Kurtosis
   L= len(IntrinsicMFs[p-1])
   features_test[j,6] = sum(np.power(IntrinsicMFs[p-1]-np.mean(IntrinsicMFs[p-1]),4)) / ((L-1)*np.power(features_test[j,3],4))

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

    
    