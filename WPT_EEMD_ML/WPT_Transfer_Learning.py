
import time
import numpy as np
import scipy.io as sio
from scipy.stats import skew
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import os
from matplotlib import rc
import matplotlib
matplotlib.rcParams.update({'font.size': 14})
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


#%% Transfer learning application which trains on one dataset and test on another one
    
def WPT_Transfer_Learning(stickout_length_training, stickout_length_test, WPT_Level, Classifier):
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
    
    :param int (WPT_Level): 
        Level of Wavelet Packet Decomposition
    
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
                       
           >>> from WPT_Transfer_Learning import WPT_Transfer_Learning
          
           #parameters
            
           >>> stickout_length_training = '2'
           >>> stickout_length_test = '4p5'
           >>> WPT_Level=4
           >>> Classifier='SVC'
         
           >>> results = WPT_Transfer_Learning(stickout_length_training, 
           >>>                                     stickout_length_test,
           >>>                                    WPT_Level, Classifier)     
           Enter the path of training set data files:
           >>> D\...\cutting_tests_processed\data_2inch_stickout
           Enter the path of test set data files:
           >>> D\...\cutting_tests_processed\data_4p5inch_stickout

    """
    #%% get the path to data files from user
    
    user_input_train = input("Enter the path of training set data files: ")
    
    assert os.path.exists(user_input_train), "Specified file does not exist at, "+str(user_input_train)
    
    user_input_test = input("Enter the path of test set data files: ")
    
    assert os.path.exists(user_input_test), "Specified file does not exist at, "+str(user_input_test)
    
    
    folderToLoad1 = os.path.join(user_input_train)
    folderToLoad2 = os.path.join(user_input_test)
    
    #%% start timer
    start2 = time.time()
        
    #%% Loading time series and labels of the classification
        
    #training set data files-------------------------------------------------------
    # import the list including the name of the time series of the chosen case
    file_name_training = 'time_series_name_'+stickout_length_training+'inch.txt'
    file_path_training = os.path.join(folderToLoad1, file_name_training)
    f = open(file_path_training,'r',newline='\n')
    
    #save the time series name into a list
    namets_training = []
    for line in f:
        names = line.split("\r\n")
        namets_training.append(names[0])
        
    #import the classification labels
    label_file_name = stickout_length_training+'_inch_Labels_2Class.npy'
    file_path1 = os.path.join(folderToLoad1, label_file_name)
    label_training = np.load(file_path1)
    
    #test set data files-----------------------------------------------------------
    # import the list including the name of the time series of the chosen case
    file_name_test = 'time_series_name_'+stickout_length_test+'inch.txt'
    file_path_test = os.path.join(folderToLoad2, file_name_test)
    f = open(file_path_test,'r',newline='\n')
    
    #save the time series name into a list
    namets_test = []
    for line in f:
        names = line.split("\r\n")
        namets_test.append(names[0])
        
    #import the classification labels
    label_file_name = stickout_length_test+'_inch_Labels_2Class.npy'
    file_path1 = os.path.join(folderToLoad2, label_file_name)
    label_test = np.load(file_path1) 
    
    #%% Upload the Decompositions and compute the feature from them----------------
    
    
    # length of datasets
    numberofcase_train = len(namets_training)
    numberofcase_test = len(namets_test)
    
    featuremat_train= np.zeros((numberofcase_train,10))
    featuremat_test= np.zeros((numberofcase_test,10))
    
    #load datasets and compute features
    for i in range (0,numberofcase_train):
        name =  'ts_%d' %(i+1)
        nameofdata = 'WPT_Level%s_Recon_%sinch_%s' %(str(WPT_Level),stickout_length_training,namets_training[i])
        pathofdata = os.path.join(folderToLoad1, nameofdata)
        ts = sio.loadmat(pathofdata)
        ts= ts["recon"]
    
        featuremat_train[i,0] = np.average(ts)
        featuremat_train[i,1] = np.std(ts)
        featuremat_train[i,2] = np.sqrt(np.mean(ts**2))   
        featuremat_train[i,3] = max(abs(ts))
        featuremat_train[i,4] = skew(ts)
        L=len(ts)
        featuremat_train[i,5] = sum(np.power(ts-featuremat_train[i,0],4)) / ((L-1)*np.power(featuremat_train[i,1],4))
        featuremat_train[i,6] = featuremat_train[i,3]/featuremat_train[i,2]
        featuremat_train[i,7] = featuremat_train[i,3]/np.power((np.average(np.sqrt(abs(ts)))),2)
        featuremat_train[i,8] = featuremat_train[i,2]/(np.average((abs(ts))))
        featuremat_train[i,9] = featuremat_train[i,3]/(np.average((abs(ts)))) 
    
    for i in range (0,numberofcase_test):
        name =  'ts_%d' %(i+1)
        nameofdata = 'WPT_Level%s_Recon_%sinch_%s' %(str(WPT_Level),stickout_length_test,namets_test[i])
        pathofdata = os.path.join(folderToLoad2, nameofdata)
        ts = sio.loadmat(pathofdata)
        ts= ts["recon"]
    
        featuremat_test[i,0] = np.average(ts)
        featuremat_test[i,1] = np.std(ts)
        featuremat_test[i,2] = np.sqrt(np.mean(ts**2))   
        featuremat_test[i,3] = max(abs(ts))
        featuremat_test[i,4] = skew(ts)
        L=len(ts)
        featuremat_test[i,5] = sum(np.power(ts-featuremat_test[i,0],4)) / ((L-1)*np.power(featuremat_test[i,1],4))
        featuremat_test[i,6] = featuremat_test[i,3]/featuremat_test[i,2]
        featuremat_test[i,7] = featuremat_test[i,3]/np.power((np.average(np.sqrt(abs(ts)))),2)
        featuremat_test[i,8] = featuremat_test[i,2]/(np.average((abs(ts))))
        featuremat_test[i,9] = featuremat_test[i,3]/(np.average((abs(ts)))) 
    
    #%% load frequency domain features (At different levels of WPT) and combine them 
    #   with the time domain feature
    
    n_feature=14 
        
    #training set------------------------------------------------------------------
    freq_feature_file_name = 'WPT_Level%d_Freq_Features_%sinch.mat'%(WPT_Level,stickout_length_training)
    file_path_Ff = os.path.join(folderToLoad1, freq_feature_file_name)        
    freq_features = sio.loadmat(file_path_Ff)
    freq_features = freq_features['Freq_Features']
    
    #concatanate the frequency and time domain features 
    featuremat_train = np.concatenate((featuremat_train, freq_features),axis = 1)
    
    #test set----------------------------------------------------------------------
    freq_feature_file_name = 'WPT_Level%d_Freq_Features_%sinch.mat'%(WPT_Level,stickout_length_test)
    file_path_Ff = os.path.join(folderToLoad2, freq_feature_file_name)        
    freq_features = sio.loadmat(file_path_Ff)
    freq_features = freq_features['Freq_Features']
    
    #concatanate the frequency and time domain features 
    featuremat_test = np.concatenate((featuremat_test, freq_features),axis = 1)
    
    #%%
    #creating train, test, accuracy, meanscore and deviation matrices
    
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
        F_Training_Train,F_Training_Test,Label_Training_Train,Label_Training_Test= train_test_split(featuremat_train, label_training, test_size=0.33)
        F_Test_Train,F_Test_Test,Label_Test_Train,Label_Test_Test= train_test_split(featuremat_test,label_test, test_size=0.70)
        
        #classifier
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
        
        #create a list that contains index number of ranked features
        rankedlist = np.zeros((14,1))
            
        #finding index of the ranked features and creating new training and test sets with respect to this ranking
        for m in range (1,15):
            k=np.where(rank==m)
            rankedlist[m-1]=k[0][0]
            F_Training_Train[:,m-1] = F_Training_Train[:,int(rankedlist[m-1][0])]
            F_Test_Test[:,m-1] = F_Test_Test[:,int(rankedlist[m-1][0])] 
        RankedList.append(rankedlist)
        
        #trying various combinations of ranked features such as ([1],[1,2],[1,2,3]...)
        for p in range(0,14): 
            start1 = time.time()
            clf.fit(F_Training_Train[:,0:p+1],Label_train)
            score1=clf.score(F_Test_Test[:,0:p+1],Label_test)
            score2=clf.score(F_Training_Train[:,0:p+1],Label_train)
            accuracy1[p,o]=score1
            accuracy2[p,o]=score2
            end1=time.time()
            duration1[p,o] = end1 - start1
    
    #computing mean score and deviation for each combination tried above        
    for n in range(0,14):
        deviation1[n,0]=np.std(accuracy1[n,:])
        deviation2[n,0]=np.std(accuracy2[n,:])
        meanscore1[n,0]=np.mean(accuracy1[n,:])
        meanscore2[n,0]=np.mean(accuracy2[n,:])
        meanduration[n,0]=np.mean(duration1[n,:])
        
    
    results = np.concatenate((meanscore1,deviation1,meanscore2,deviation2),axis=1)
    results = 100*results    
    
    #total duration for algorithm  
    end2 = time.time()
    duration2 = end2-start2
    
    
    return results,print('Total elapsed time: {}'.format(duration2)) ,featuremat_train, featuremat_test  
    