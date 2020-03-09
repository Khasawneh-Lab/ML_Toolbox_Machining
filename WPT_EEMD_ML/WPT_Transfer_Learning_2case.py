"""
Transfer Learning On Two Cases Using WPT
----------------------------------------

This fuction implement transfer learning by training a classifier on two different 
data sets and testing it on remaining two different data sets. Stickout lengths of 
each data set should be determined by user. It asks for file paths to training and test set files. 
It assumes that reconstructed wavelet packets and frequency domain features are available in the specified data file folder.
Algorithm performs classfication based on chosen classifier and returns the results in an array. 
It also prints the total elapsed time.  

"""
import time
import numpy as np
import scipy.io as sio
from scipy.stats import skew
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from matplotlib import rc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib
import os
matplotlib.rcParams.update({'font.size': 14})
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
#%%
def WPT_Transfer_Learning_2case(stickout_lengths,WPT_Level,Classifier):
    """
    :param str (stickout_lengths): 
       Stickout length for the training and test set in a np.array([]) format.First two stickout length are considered as training set data and the remaining ones are test set data.
       
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
                       
           >>> from WPT_Transfer_Learning_2case import WPT_Transfer_Learning_2case
          
           #parameters
            
           >>> stickout_lengths = ['2','2p5','3p5','4p5']
           >>> WPT_Level=4
           >>> Classifier='SVC'
         
           >>> results = WPT_Transfer_Learning_2case(stickout_lengths, 
           >>>                                       WPT_Level, Classifier)     
           Enter the path of first training set data files:
           >>> D\...\cutting_tests_processed\data_2inch_stickout
           Enter the path of second training set data files:
           >>> D\...\cutting_tests_processed\data_2p5inch_stickout
           Enter the path of first test set data files:
           >>> D\...\cutting_tests_processed\data_3p5inch_stickout   
           Enter the path of second test set data files:
           >>> D\...\cutting_tests_processed\data_4p5inch_stickout
    """    
    #%% parameters
#    stickout_length_training1 = '2'
#    stickout_length_training2 = '2'
#    stickout_length_test1 = '2'
#    stickout_length_test2 = '2'
#    
#    WPT_Level=1
#    Classifier = 'SVC'
##    
#    stickout_lengths = np.array([stickout_length_training1,stickout_length_training2, stickout_length_test1, stickout_length_test2])
    
    #%% get the path to data files from user
    
    user_input_train1 = input("Enter the path of first training set data files: ")
    
    assert os.path.exists(user_input_train1), "Specified file does not exist at, "+str(user_input_train1)
    
    user_input_train2 = input("Enter the path of second training set data files: ")
    
    assert os.path.exists(user_input_train2), "Specified file does not exist at, "+str(user_input_train2)
    
    user_input_test1 = input("Enter the path of first test set data files: ")
    
    assert os.path.exists(user_input_test1), "Specified file does not exist at, "+str(user_input_test1)
    
    user_input_test2 = input("Enter the path of second test set data files: ")
    
    assert os.path.exists(user_input_test2), "Specified file does not exist at, "+str(user_input_test2)
    
    folderToLoad1 = os.path.join(user_input_train1)
    folderToLoad2 = os.path.join(user_input_train2)
    folderToLoad3 = os.path.join(user_input_test1)
    folderToLoad4 = os.path.join(user_input_test2)
    
    #%% start timer
    start2 = time.time()
    
    #%% load data files compute time domain features and combine them with frequency domain features computed in Matlab previously
    n_feature=14
    # import the list including the name of the time series of the chosen case
    label={}
    feature_mat={}
    for i in range(0,4):
        if i==0:
            folderToLoad = folderToLoad1
        elif i==1:
            folderToLoad = folderToLoad2 
        elif i==2:
            folderToLoad = folderToLoad3
        elif i==3:
            folderToLoad = folderToLoad4   
            
        file_name = 'time_series_name_'+stickout_lengths[i]+'inch.txt'
#        folderToLoad_name = 'folderToLoad'+str(i+1)
#        exec("folderToLoad = %s" %(folderToLoad_name),globals())
        file_path = os.path.join(folderToLoad, file_name)
        f = open(file_path,'r',newline='\n')
        
        #save the time series name into a list
        namets = []
        for line in f:
            names = line.split("\r\n")
            namets.append(names[0])
    
        #import the classification labels
        label_file_name = stickout_lengths[i]+'_inch_Labels_2Class.npy'
        label_path = os.path.join(folderToLoad, label_file_name)
        label[i] = np.load(label_path) 
      
    
        # load datasets
        numberofcase = len(namets)
        feature_mat[i]=np.zeros((numberofcase,10))
                
        for j in range(numberofcase):
            nameofdata = 'WPT_Level%s_Recon_%sinch_%s' %(str(WPT_Level),stickout_lengths[i],namets[j])
            pathofdata = os.path.join(folderToLoad, nameofdata)
            ts = sio.loadmat(pathofdata)
            ts= ts["recon"]
            
            feature_mat[i][j,0] = np.average(ts)
            feature_mat[i][j,1] = np.std(ts)
            feature_mat[i][j,2] = np.sqrt(np.mean(ts**2))
            feature_mat[i][j,3] = max(abs(ts))
            feature_mat[i][j,4] = skew(ts)
            L=len(ts)
            feature_mat[i][j,5] = sum(np.power(ts-feature_mat[i][j,0],4)) / ((L-1)*np.power(feature_mat[i][j,1],4))
            feature_mat[i][j,6] = feature_mat[i][j,3]/feature_mat[i][j,2]
            feature_mat[i][j,7] = feature_mat[i][j,3]/np.power((np.average(np.sqrt(abs(ts)))),2)
            feature_mat[i][j,8] = feature_mat[i][j,2]/(np.average((abs(ts))))
            feature_mat[i][j,9] = feature_mat[i][j,3]/(np.average((abs(ts))))
            
        freq_feature_file_name = 'WPT_Level%d_Freq_Features_%sinch.mat'%(WPT_Level,stickout_lengths[i])
        file_path_Ff = os.path.join(folderToLoad, freq_feature_file_name)        
        freq_features = sio.loadmat(file_path_Ff)
        freq_features = freq_features['Freq_Features']
         
        #concatanate the frequency and time domain features 
        feature_mat[i]=np.concatenate((feature_mat[i], freq_features),axis = 1)
        
        
    #concatanate the frequency and time domain features 
    featuremat_training  =  np.concatenate((feature_mat[0], feature_mat[1]),axis = 0)
    featuremat_test  = np.concatenate((feature_mat[2], feature_mat[3]),axis = 0)
    
    # labels of training set and test sets
    label_train = np.concatenate((label[0],label[1]),axis=0)
    label_test = np.concatenate((label[2],label[3]),axis=0)
    
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
        F_Training_Train,F_Training_Test,Label_Training_Train,Label_Training_Test= train_test_split(featuremat_training, label_train, test_size=0.33)
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
    
    
    return results,print('Total elapsed time: {}'.format(duration2)) ,featuremat_training, featuremat_test  

    # This part of the code includes the ranked features for each iteration and keep them in arrays
    
    #how_many_times_rank = np.zeros((14,14))
    #for i in range (0,14):
    #    for j in range(0,10):
    #        a = RankedList[j][i][0]
    #        a = int(a)
    #        how_many_times_rank[a,i]=how_many_times_rank[a,i]+1
    #
    #sio.savemat('number_of_times_feature_ranks_4.5inch_WPT_Level4.mat',mdict={'times_feature_rank':how_many_times_rank})

