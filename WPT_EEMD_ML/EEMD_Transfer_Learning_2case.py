# -*- coding: utf-8 -*-
"""
Transfer Learning On Two Cases Using EEMD
-----------------------------------------

This code takes four different stickout cases in an array from user. It uses first two cases as training set samples and remaining two cases are considered as 
test set data sets. 
Algorithm assumes that the IMFs for EEMD have alredy been computed and saved.
The file paths to the IMFs and data files should be included in 'file_paths.txt' file and this file should be in the same folder with this function.
This function generates the feature matrices for training and test correponding to given stickout cases and their informative Intrinsic Mode Functions (IMFs). 
It perform classification based on classifier name specified by the user.
It returns the mean accuracies and deviations in a np.array([]). It also prints the total elapsed time.

"""

import time
start = time.time()
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

def EEMD_Transfer_Learning_2case(stickout_lengths,IMF_Number,Classifier):
    """
    :param str (file_path.txt): 
        Algorithm needs .txt file whose name is 'file_paths.txt' to add the paths of the data files and their corresponding decompositions. This .txt file should include 8 file path, four for data files and four for decompositions.First four file paths are for data files (time series). Other four are considered as paths to the decompositions. Each line in .txt file should include only one path. The ordering for the paths is such that the first path represents first case of training set data files. Second path is for the second case of training set data files. Third and fourth paths represent first test set case and second test set case data files. Same ordering should be followed for the paths of decompositions.
    :param str (stickout_lengths): 
       Training and test set cases in a np.array([]). The algorithm considers the first two entry as training set cases and remanining ones as test set cases.

       * if stickout length is 2 inch, '2'
       * if stickout length is 2.5 inch, '2p5'
       * if stickout length is 3.5 inch, '3p5'
       * if stickout length is 4.5 inch, '4p5'
      
    :param int (IMF_Number): 
        Corresponding Informative Intrinsic Mode Function (IMF) numbers in an array for cases given in stickout_lengths. First two entry are considered as training set IMFs and remaining ones are for test set IMFs.
    
    :param str (Classifier): 
       Classifier defined by user
       
       * Support Vector Machine: 'SVC'
       * Logistic Regression: 'LR'
       * Random Forest Classification: 'RF'
       * Gradient Boosting: 'GB'
    
    :Returns:

        :results:
            (np.array([])) Classification results for training and test set for all combination of ranked features and devition for both set.
        
            * first column: mean accuracies for training set
            * second column: deviation for training set accuracies
            * third column: mean accuracies for test set
            * fourth column: deviation for test set accuracies
        
        :time:
            (str) Elapsed time during feature matrix generation and classification
    
    :Example:
    
        .. doctest::
                       
           >>> from EEMD_Transfer_Learning_2case import EEMD_Transfer_Learning_2case
          
           #parameters
            
           >>> stickout_lengths = ['2','2p5','3p5','4p5']
           >>> IMF_Number=[2,2,1,1]
           >>> Classifier = 'RF'
    
           >>> results = EEMD_Transfer_Learning_2case(stickout_lengths,IMF_Number,Classifier)

       """    
#%% add paths to the decompositions (if available) and data files
    file_name = 'file_paths.txt'
    f = open(file_name,'r',newline='\n')
    paths = []
    for line in f:
        names = line.split("\r\n")
        paths.append(names[0])   
        
    #%% 
    foldersToLoad = {}
    for i in range(8):
        foldersToLoad[i] = os.path.join(paths[i])
    
    # start timer 
    start =time.time() 
      #%% Loading time series and labels of the classification
    namets={}
    rpm = {}
    doc = {}
    label = {}
    
    for k in range(4):
        stickout_length = stickout_lengths[k]
        folderToLoad = foldersToLoad[k]
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
    C_D_Divided_={}
    CaseLabels ={}
    infoEMF={}
    
    for k in range(4):
        stickout_length = stickout_lengths[k]
        folderToLoad = foldersToLoad[k]
        numberofcase = len(namets[k])
        
        ts={}
        #load datasets
        for i in range (0,numberofcase):
            nameofdata = '%s' %(namets[k][i])
            pathofdata = os.path.join(folderToLoad, nameofdata)
            time_s = sio.loadmat(pathofdata)
            ts[i] = time_s["tsDS"]
        
        #labeled and concatanated matrix for first dataset
        label1=np.full((len(ts[0]),1),rpm[k][0])
        label2=np.full((len(ts[0]),1),doc[k][0])
        label3=np.full((len(ts[0]),1),label[k][0])
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
            labelrpm=np.full((L,1),rpm[k][i+1])
            labeldoc=np.full((L,1),doc[k][i+1])
            label_c=np.full((L,1),label[k][i+1])
            chatter_data.append(data)
            labels=np.concatenate((labelrpm,labeldoc,label_c),axis=1)
            case_label.append(labels)
        
        N=len(chatter_data)   #length of actual cases
        C_D = chatter_data
            
        #length of each case
        length=np.zeros((N,1))
        for i in range(0,N):
            length[i]=len(C_D[i])
        
        
        
        caseLabels = np.zeros((1,3))  #initialize the matrix for labels
        
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
        C_D_Divided_[k]=C_D_Divided[0:inc]
        
        case = np.zeros((inc,1))
        for i in range(0,inc):
            case[i]=i
        
        CaseLabels[k]=np.concatenate((caseLabels,case),axis=1)
        
        infoEMF[k]=np.ndarray(shape=(len(C_D_Divided_[k])),dtype=object)
    
    label_training1 = CaseLabels[0][:,2]
    label_training2 = CaseLabels[1][:,2]
    label_test1 = CaseLabels[2][:,2]
    label_test2 = CaseLabels[3][:,2]
    
    #%% load eIMFs if they are computed before
    infoEMFs={}
    for k in range(4):
        sys.path.insert(0,foldersToLoad[k+4])
        lengthofcases = len(C_D_Divided_[k])
        infoIMFs=np.ndarray(shape=(lengthofcases),dtype=object)
        for i in range(0,lengthofcases):
            dataname = 'IMFs_%sinch_Divided_Data_IMFs_Case%d' %(stickout_lengths[k],i+1) 
            Decomp = sio.loadmat(os.path.join(foldersToLoad[k+4], dataname))
            infoIMFs[i]= Decomp['eIMF']
        infoEMFs[k]=infoIMFs
    
    #%% compute features for eIMFs
    features_ = {}
    for m in range(4):    
        p = IMF_Number[m]
        eIMFs_ = infoEMFs[m]
    
        features=np.zeros((len(C_D_Divided_[m]),7))
        for i in range(0,len(C_D_Divided_[m])):
            eIMFs = eIMFs_[i]
            #feature_1
            nIMFs=len(eIMFs)
            A = np.power(eIMFs[p-1],2) 
            A_sum = sum(A)                                   #summing squares of whole elements of second IMF
            B_sum = 0               
            for k in range(nIMFs):
                B_sum = B_sum + sum(np.power(eIMFs[k],2))   #computing summing of squares of whole elements of IMFs
            features[i,0]=A_sum/B_sum                        #energy ratio feature
            
            #feature_2  Peak to peak value
            Maximum = max(eIMFs[p-1])
            Minimum = min(eIMFs[p-1])
            features[i,1] = Maximum - Minimum 
            #feature_3 standard deviation
            features[i,2] = np.std(eIMFs[p-1])
            #feature_4 root mean square (RMS)
            features[i,3] = np.sqrt(np.mean(eIMFs[p-1]**2))   
            #feature_5 Crest factor
            features[i,4] = Maximum/features[i,3]
            #feature_6 Skewness
            features[i,5] = skew(eIMFs[p-1])
            #feature_7 Kurtosis
            L= len(eIMFs[p-1])
            features[i,6] = sum(np.power(eIMFs[p-1]-np.mean(eIMFs[p-1]),4)) / ((L-1)*np.power(features[i,3],4))
        features_[m] = features
                
    
    #%% CONCETANETE THE FEATURE MATRICES AND THE LABELS FOR TRAINING SET AND TEST SET
       
    features_training = np.concatenate((features_[0],features_[1]),axis=0)
    features_test =  np.concatenate((features_[2],features_[3]),axis=0)
    
    label_training = np.concatenate((label_training1,label_training2),axis=0)
    label_test = np.concatenate((label_test1,label_test2),axis=0)
    
    #%%---------------CLASSIFICATION-----------------------------------------------
  
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
        if Classifier=='SVC':
            clf = SVC(kernel='linear')
        elif Classifier=='LR':
            clf = LogisticRegression()
        elif Classifier=='RF':
            clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        elif Classifier=='GB':
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
        
        #find index of the ranked features and creating new training and test sets with respect to this ranking
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

    return results
    
    
