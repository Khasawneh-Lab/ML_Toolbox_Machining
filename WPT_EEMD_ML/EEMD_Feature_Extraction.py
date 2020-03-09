"""
Feature extraction and supervised classification using EEMD 
-----------------------------------------------------------

This function takes time series for turning cutting tests as inputs.
It ask user to enter the file paths for the data files. If user specify that the decompositions for EEMD have already been computed, algorithm will ask user to enter the file paths for the decompositions.
Otherwise, it will compute the IMFs and ask users to enter the file paths where they want to save these decompositions.
Based on given stickout length cases and corresponding informative IMF numbers, it will generate the feature matrix and perform the classification with specified classification algorithm by user.
It returns the results in an array and prints the total elapsed time.

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


def EEMD_Feature_Extraction(stickout_length, EEMDecs, p, Classifier):
    """
    
    :param str (stickout_lengths): The distance between heel of the boring bar and the back surface of the cutting tool 
    
       * if stickout length is 2 inch, '2'
       * if stickout length is 2.5 inch, '2p5'
       * if stickout length is 3.5 inch, '3p5'
       * if stickout length is 4.5 inch, '4p5'
    
    :param str (EEMDecs): 
       
       * if decompositions have already been computed, 'A'
       * if decompositions have not been computed, 'NA'    
       
    :param int (p): Informative intrinsic mode function (IMF) number
  
    :param str (Classifier): Classifier defined by user
       
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
                       
           >>> from EEMD_Feature_Extraction import EEMD_Feature_Extraction
            
           #parameters
           >>> stickout_length='2'
           >>> EEMDecs = 'A'
           >>> p=2
           >>> Classifier = 'SVC'
         
           >>> results = EEMD_Feature_Extraction(stickout_length, EEMDecs, 
           >>>                                  p, Classifier)     
           Enter the path of the data files:
           >>> D\...\cutting_tests_processed\data_2inch_stickout
           Enter Enter the path of EEMD files:
           >>> D\...\eIMFs\data_2inch_stickout
           
    """

    #%%
    user_input = input("Enter the path of the data files: ")
    
    assert os.path.exists(user_input), "Specified file does not exist at, "+str(user_input)
    
    folderToLoad = os.path.join(user_input)
    
    if EEMDecs == 'A':
        user_input2 = input("Enter the path of EEMD files: ")
    
        assert os.path.exists(user_input2), "Specified file does not exist at, "+str(user_input2)
    
        folderToLoad2 = os.path.join(user_input2)
        
    if EEMDecs == 'NA':
        user_input3 = input("Enter the path to save EEMD files: ")
    
        assert os.path.exists(user_input3), "Specified file does not exist at, "+str(user_input3)
    
        folderToLoad3 = os.path.join(user_input3)   
    
    # start timer 
    start4 =time.time() 
        
    #%% Loading time series and labels of the classification
    
    # import the list including the name of the time series of the chosen case
    file_name = 'time_series_name_'+stickout_length+'inch.txt'
    file_path = os.path.join(folderToLoad, file_name)
    f = open(file_path,'r',newline='\n')
    
    #save the time series name into a list
    namets = []
    for line in f:
        names = line.split("\r\n")
        namets.append(names[0])
    
    file_name = 'time_series_rpm_'+stickout_length+'inch.txt'
    file_path = os.path.join(folderToLoad, file_name)
    f = open(file_path,'r',newline='\n')
    
    #save the time series name into a list
    rpm = []
    for line in f:
        rpms = line.split("\r\n")
        rpm.append(int(rpms[0]))
    rpm=np.asarray(rpm)
    
    
    file_name = 'time_series_doc_'+stickout_length+'inch.txt'
    file_path = os.path.join(folderToLoad, file_name)
    f = open(file_path,'r',newline='\n')
    
    #save the time series name into a list
    doc = []
    for line in f:
        docs = line.split("\r\n")
        doc.append(float(docs[0]))
    doc=np.asarray(doc)
    
    
    #import the classification labels
    label_file_name = stickout_length+'_inch_Labels_2Class.npy'
    file_path1 = os.path.join(folderToLoad, label_file_name)
    label = np.load(file_path1)
    
        
    #%% Upload the Decompositions and compute the feature from them----------------
    #name of datasets
    
    numberofcase = len(namets)     
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
    
    
    #%% Compute IMFs if they are not computed before
    if EEMDecs=='NA':
        from PyEMD import EEMD
        eemd = EEMD()
        emd = eemd.EMD
        emd.trials = 200      #default = 100
        emd.noise_width = 0.2 #default = 0.05
        
        infoEMF=np.ndarray(shape=(len(C_D_Divided)),dtype=object)
    
        #EEMD
        #chosen imf for feature extraction 
    
        for i in range(0,len(C_D_Divided)): 
        
            #signal
            S = C_D_Divided[i][:,1]
            t = C_D_Divided[i][:,0]
            
            eIMFs = emd(S, t)
            nIMFs = eIMFs.shape[0]
            infoEMF[i]=eIMFs  
            print('Progress: IMFs were computed for case number {}.  '.format(i))
    
            #save eIMFs into mat file 
            name = 'IMFs_'+stickout_length+'inch_Divided_Data_IMFs_Case%i.mat'%(i+1)
            save_name = folderToLoad3+'\\'+name
            sio.savemat(save_name,{'eIMF':infoEMF[i]})
            
    #%% load eIMFs if they are computed before
    if EEMDecs=='A':
     
    #create a path to file including the IMFs
        sys.path.insert(0,folderToLoad2)
        for i in range(0,len(C_D_Divided)):
            dataname = 'IMFs_%sinch_Divided_Data_IMFs_Case%d' %(stickout_length,i+1) 
            infoEMF[i] = sio.loadmat(os.path.join(folderToLoad2, dataname))
            infoEMF[i] = infoEMF[i]['eIMF']
    
    
    
    #%% compute features for eIMFs
    features=np.zeros((len(C_D_Divided),7))
    for i in range(0,len(C_D_Divided)):
        eIMFs = infoEMF[i]
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
     
    #%% classification
    n_feature=7
    
    #generating accuracy, meanscore and deviation matrices
    split1,split2 = train_test_split(features, test_size=0.33)
    F_traincomb = np.zeros((len(split1),7))
    F_testcomb = np.zeros((len(split2),7))
    
    
    accuracy1 = np.zeros((n_feature,10))
    accuracy2 = np.zeros((n_feature,10))
    deviation1 = np.zeros((n_feature,1))
    deviation2 = np.zeros((n_feature,1))
    meanscore1 = np.zeros((n_feature,1))
    meanscore2 = np.zeros((n_feature,1))
    duration1 = np.zeros((n_feature,10))
    meanduration = np.zeros((n_feature,1))
    
    #repeat the procedure ten times 
    Rank=[]
    RankedList=[]
    
    for o in range(0,10):
        
        #split into test and train set
        F_train,F_test,Label_train,Label_test= train_test_split(features,caseLabels, test_size=0.33)
        
        #Labels
        Label_train = Label_train[:,2]
        Label_test = Label_test[:,2]
        
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
        selector = selector.fit(F_train, Label_train)
        rank = selector.ranking_
        Rank.append(rank)
        rank = np.asarray(rank)
        
        #create a list that contains index numbe of ranked features
        rankedlist = np.zeros((n_feature,1))
    
        
        #finding index of the ranked features and creating new training and test sets with respect to this ranking
        for m in range (1,n_feature+1):
            k=np.where(rank==m)
            rankedlist[m-1]=k[0][0]
            F_traincomb[:,m-1] = F_train[:,int(rankedlist[m-1][0])]
            F_testcomb[:,m-1] = F_test[:,int(rankedlist[m-1][0])] 
        RankedList.append(rankedlist)
        
        #trying various combinations of ranked features such as ([1],[1,2],[1,2,3]...)
        for p in range(0,n_feature): 
            start1 = time.time()
            clf.fit(F_traincomb[:,0:p+1],Label_train)
            score1=clf.score(F_testcomb[:,0:p+1],Label_test)
            score2=clf.score(F_traincomb[:,0:p+1],Label_train)
            accuracy1[p,o]=score1
            accuracy2[p,o]=score2
            end1=time.time()
            duration1[p,o] = end1 - start1
    
    #computing mean score and deviation for each combination tried above        
    for n in range(0,n_feature):
        deviation1[n,0]=np.std(accuracy1[n,:])
        deviation2[n,0]=np.std(accuracy2[n,:])
        meanscore1[n,0]=np.mean(accuracy1[n,:])
        meanscore2[n,0]=np.mean(accuracy2[n,:])
        meanduration[n,0]=np.mean(duration1[n,:])
        
    results = np.concatenate((meanscore1,deviation1,meanscore2,deviation2),axis=1)
    results = 100*results    
        
    #total duration for algorithm  
    end4 = time.time()
    duration4 = end4-start4
    print('Total elapsed time: {} seconds.'.format(duration4))
    return results, print('Total elapsed time: {}'.format(duration4)),features  
    