"""
Transfer Learning Application Using EEMD
----------------------------------------
This function uses transfer learning principles on two differen stickout cases for cutting data set. 
It uses one of the cases as training set and train a specified classifier on this data set, then it tests the classfier on the other cases provided by the user.
This functions assumes that the IMFs for EEMD have alread been computed and it asks the paths for data files and decompositions (IMFs).
The informative IMF numbers should be provided by user for training and test set seperately. 
Then, function will generate feature matrix and perform classfication with chosen algorithm by user.
It returns the results in a np.array and prints the total elapsed time.

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

def EEMD_Transfer_Learning(stickout_length_training, stickout_length_test, p_train, p_test, Classifier):
    """
    :param str (stickout_length_training): 
       Stickout length for the training data set 
       
       * if stickout length is 2 inch, '2'
       * if stickout length is 2.5 inch, '2p5'
       * if stickout length is 3.5 inch, '3p5'
       * if stickout length is 4.5 inch, '4p5'
       
    :param str (stickout_length_test): 
       Stickout length for the test data set

       * if stickout length is 2 inch, '2'
       * if stickout length is 2.5 inch, '2p5'
       * if stickout length is 3.5 inch, '3p5'
       * if stickout length is 4.5 inch, '4p5'
       
    :param int (p_train): 
       Informative intrinsic mode function (IMF) number for training set 
       
    :param int (p_test): 
       Informative intrinsic mode function (IMF) number for test set 
  
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
           >>> stickout_length_training='2'
           >>> stickout_length_test='4p5'
           >>> p_train = 2
           >>> p_test = 1
           >>> Classifier = 'GB'
         
           >>> results = EEMD_Transfer_Learning(stickout_length_training, 
                                                stickout_length_test, 
           >>>                                  p_train, p_test,
                                                Classifier)     
           Enter the path of training data files:
           >>> D\...\cutting_tests_processed\data_2inch_stickout
           Enter the path of test data files:
           >>> D\...\cutting_tests_processed\data_4p5inch_stickout
           Enter the path to decompositions for training set:
           >>> D\...\eIMFs\data_2inch_stickout
           Enter the path to decompositions for test set:
           >>> D\...\eIMFs\data_4p5inch_stickout  
    """


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
    
    
    user_input4 = input("Enter the path to decompositions for test set: ")
    
    assert os.path.exists(user_input4), "Specified path does not exist at, "+str(user_input4)
    
    folderToLoad4 = os.path.join(user_input4)   
    
    # start timer 
    start =time.time() 
        
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
    
    C_D_Divided_={}
    CaseLabels ={}
    infoEMF={}
    
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
            labelrpm=np.full((L,1),rpm[k][i])
            labeldoc=np.full((L,1),doc[k][i])
            label_c=np.full((L,1),label[k][i])
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
        C_D_Divided_[k]=C_D_Divided[0:inc]
        
        case = np.zeros((inc,1))
        for i in range(0,inc):
            case[i]=i
        
        CaseLabels[k]=np.concatenate((caseLabels,case),axis=1)
        
        infoEMF[k]=np.ndarray(shape=(len(C_D_Divided_[k])),dtype=object)
    
    label_training = CaseLabels[0][:,2]
    label_test = CaseLabels[1][:,2]
    
    infoEMF_training = infoEMF[0]
    infoEMF_test = infoEMF[1]
    
    #%% load eIMFs 
    
    #training set
    sys.path.insert(0,folderToLoad3)
    for i in range(0,len(C_D_Divided_[0])):
        dataname = 'IMFs_%sinch_Divided_Data_IMFs_Case%d' %(stickout_length_training,i+1) 
        infoEMF_training[i] = sio.loadmat(os.path.join(folderToLoad3, dataname))
        infoEMF_training[i] = infoEMF_training[i]['eIMF']
    
    #test set
    sys.path.insert(0,folderToLoad4)
    for i in range(0,len(C_D_Divided_[1])):
        dataname = 'IMFs_%sinch_Divided_Data_IMFs_Case%d' %(stickout_length_test,i+1) 
        infoEMF_test[i] = sio.loadmat(os.path.join(folderToLoad4, dataname))
        infoEMF_test[i] = infoEMF_test[i]['eIMF']
    
        
        #%% compute features for eIMFs
    features_ = {}
    for m in range(2):    
        if m==0:
            p = p_train
            eIMFs_ = infoEMF_training
        else: 
            p = p_test
            eIMFs_ = infoEMF_test
            
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
        F_Training_Train,F_Training_Test,Label_Training_Train,Label_Training_Test= train_test_split(features_[0],label_training, test_size=0.33)
        F_Test_Train,F_Test_Test,Label_Test_Train,Label_Test_Test= train_test_split(features_[1],label_test, test_size=0.70)    
        
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

    return results, features_[0],features_[1],RankedList
    
    