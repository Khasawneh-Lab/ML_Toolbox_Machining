"""
Transfer Learning Application Using WPT
---------------------------------------

This fuction uses tranfer learning principles to diagnose chatter in time series 
obtained from turning cutting experiment. Intrinsic mode functions (IMFs) or decomposition 
for each time series should have been computed to be able to use this code.   

"""
import time
start2 = time.time()
import numpy as np
import pandas as pd
import scipy.io as sio
import os.path
import sys
from scipy.stats import skew
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier




#%% parameters:
stickout_length_training='2'
stickout_length_test='4p5'


#%%
user_input = input("Enter the path of training data files: ")

assert os.path.exists(user_input), "Specified path does not exist at, "+str(user_input)

folderToLoad1 = os.path.join(user_input)


user_input2 = input("Enter the path of test data files: ")

assert os.path.exists(user_input2), "Specified path does not exist at, "+str(user_input2)

folderToLoad2 = os.path.join(user_input2)
    

user_input3 = input("Enter the path to decompositions for training set: ")

assert os.path.exists(user_input3), "Specified path does not exist at, "+str(user_input3)

folderToLoad3 = os.path.join(user_input3)   


user_input4 = input("Enter the path to decompositions for training set: ")

assert os.path.exists(user_input4), "Specified path does not exist at, "+str(user_input4)

folderToLoad4 = os.path.join(user_input4)   

# start timer 
start4 =time.time() 
    
    #%% Loading time series and labels of the classification
namets={}
rpm = {}
doc = {}
label = {}

for k in range(2):   
    if k==0:
        stickout_length = stickout_length_training
        folderToLoad = folderToLoad1
    else:
        stickout_length = stickout_length_test
        folderToLoad = folderToLoad2
    # import the list including the name of the time series of the chosen case
    file_name = 'time_series_name_'+stickout_length+'inch.txt'
    file_path = os.path.join(folderToLoad, file_name)
    f = open(file_path,'r',newline='\n')
    
    #save the time series name into a list
    namets_ = []
    for line in f:
        names = line.split("\r\n")
        namets_.append(names[0])
    namets[k]=namets_
    
    file_name = 'time_series_rpm_'+stickout_length+'inch.txt'
    file_path = os.path.join(folderToLoad, file_name)
    f = open(file_path,'r',newline='\n')
    
    #save the time series name into a list
    rpm_ = []
    for line in f:
        rpms = line.split("\r\n")
        rpm_.append(int(rpms[0]))
    rpm_=np.asarray(rpm_)
    rpm[k]=rpm_
    
    file_name = 'time_series_doc_'+stickout_length+'inch.txt'
    file_path = os.path.join(folderToLoad, file_name)
    f = open(file_path,'r',newline='\n')
    
    #save the time series name into a list
    doc_ = []
    for line in f:
        docs = line.split("\r\n")
        doc_.append(float(docs[0]))
    doc_=np.asarray(doc_)
    doc[k]=doc_
    
    #import the classification labels
    label_file_name = stickout_length+'_inch_Labels_2Class.npy'
    file_path1 = os.path.join(folderToLoad, label_file_name)
    label_ = np.load(file_path1)
    label[k] = label_

#%% length of datasets
numberofcase1 = len(namets[0])
numberofcase2 = len(namets[1])


for k in range(2):
    if k==0:
      numberofcase = numberofcase1
      folderToLoad = folderToLoad1
    else:
      numberofcase = numberofcase2
      folderToLoad = folderToLoad2
     
    ts={}
    #load datasets and compute features
    for i in range (0,numberofcase):
        nameofdata = '%s' %(namets[i])
        pathofdata = os.path.join(folderToLoad, nameofdata)
        time_s = sio.loadmat(pathofdata)
        ts[i] = time_s["tsDS"]
    
    #labeled and concatanated matrix for first dataset
    label1=np.full((len(ts[0]),1),320)
    label2=np.full((len(ts[0]),1),0.005)
    label3=np.full((len(ts[0]),1),1)
    chatter_data=np.concatenate((ts[0],label1,label2,label3),axis=1)
    df=pd.DataFrame(chatter_data)
    
    #create concataneted dataframe in a for loop 
    chatter_data = []
    case_label = []
    
    chatter_data.append((df.values)[:,0:2])
    case_label.append(np.concatenate((label1,label2,label3),axis=1))
    
    for i in range(0,numberofcase-1):
        data=ts[i+1]
        L=len(data)
        labelrpm=np.full((L,1),rpm[i])
        labeldoc=np.full((L,1),doc[i])
        label_c=np.full((L,1),label[i])
        chatter_data.append(data)
        labels=np.concatenate((labelrpm,labeldoc,label_c),axis=1)
        case_label.append(labels)
    
    N=len(chatter_data)   #length of actual cases
    C_D = chatter_data
    
    ##feature scaling
    #if EEMD=='NA':
    #    for i in range (0,N):
    #        sc = StandardScaler()
    #        C_D[i][:,1] = sc.fit_transform(np.reshape(C_D[i][:,1],(1,-1))) #normalized data
    
    #length of each case
    length=np.zeros((N,1))
    for i in range(0,N):
        length[i]=len(C_D[i])
    
    
    
    caseLabels = np.zeros((1,3))  #intialize the matrix for labels
    
    inc = 0  # increment for total number of cases obtained after dividing
    
    approximate_number_of_cases = int((sum(length))/1000) #approximate number of cases with respect to sum of lengths of actual cases
    
    C_D_Divided=np.ndarray(shape=(approximate_number_of_cases),dtype=object)  #create object array to store new cases
    for i in range(0,N):  
        data=C_D[i]
        if len(data)>1000:
            division_number=int(len(data)/1000)            #number determines the      
            split=np.array_split(data,division_number)     #split data into different matrices not equal in size
            n=len(split)                                   #number of cases obtained from each actual case
            Label=np.reshape(case_label[i][0],(1,3))
            for j in range(0,n):
                C_D_Divided[inc]=np.array(split[j])
                caseLabels=np.append(caseLabels,Label,axis=0)
                inc=inc+1
    caseLabels=caseLabels[1:]    #delete the first row of matrix and 
    C_D_Divided=C_D_Divided[0:inc]
    
    case = np.zeros((inc,1))
    for i in range(0,inc):
        case[i]=i
    
    caseLabels=np.concatenate((caseLabels,case),axis=1)
    
    infoEMF=np.ndarray(shape=(len(C_D_Divided)),dtype=object)







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

    
    