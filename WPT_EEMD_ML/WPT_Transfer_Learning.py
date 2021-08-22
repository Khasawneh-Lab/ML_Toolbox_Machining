
import time
import numpy as np
import scipy.io as sio
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from WPT_EEMD_ML.WPT_Feature_Extraction import WPT_Feature_Extraction
import os



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
    # Compute feature matrix for training and test set cases
    
    # training set feature matrix
    data_path_train = 'D:\\Data Archive\\Cutting_Test_Data_Documented\\cutting_tests_processed\\2inch_stickout'
    list_name_train = 'time_series_name_2inch.txt'
    WF = 'db10'
    L_train=4
    IWP_train = 3
    label_name_train = '2_inch_Labels_2Class.npy'
    fs_train = 10000  
    saving = False
    feature_mat_train,label_train = WPT_Feature_Extraction(data_path_train, list_name_train,label_name_train,WF,L_train,IWP_train,fs_train,saving)    
    
    # test set feature matrix

    
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
    