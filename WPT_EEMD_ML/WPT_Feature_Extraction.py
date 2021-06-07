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
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
matplotlib.rcParams.update({'font.size': 14})
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

def WPT_Feature_Extraction(stickout_length, WPT_Level, Classifier, plotting):
    """
    
    :param str (stickout_length): The distance between heel of the boring bar and the back surface of the cutting tool 
    
       * if stickout length is 2 inch, '2'
       * if stickout length is 2.5 inch, '2p5'
       * if stickout length is 3.5 inch, '3p5'
       * if stickout length is 4.5 inch, '4p5'
    
    :param int (WPT_Level): 
        Level of Wavelet Packet Decomposition
    
    :param str (Classifier): Classifier defined by user
       
       * Support Vector Machine: 'SVC'
       * Logistic Regression: 'LR'
       * Random Forest Classification: 'RF'
       * Gradient Boosting: 'GB'
    	
    :param  str (Plotting):
        Function will return the plot of the results depending on the number of features used in the classification, if the Plotting is 'True'.
    
    :Returns:

        :results:
            (np.array([])) Classification results for training and test set for all combination of ranked features and devition for both set.
        
            * first column: mean accuracies for training set
            * second column: deviation for training set accuracies
            * third column: mean accuracies for test set
            * fourth column: deviation for test set accuracies
        
        :plot: Deviations and mean accuracies for training and test set vs number of features used in feature matrix generation
        
        :time:
            (str) Elapsed time during feature matrix generation and classification
    
    :Example:
    
        .. doctest::
                       
           >>> from WPT_Feature_Extraction import WPT_Feature_Extraction
           >>> import matplotlib.pyplot as plt
           >>> from matplotlib import rc
           >>> import matplotlib
           >>> matplotlib.rcParams.update({'font.size': 14})
           >>> rc('font',**{'family':'serif','serif':['Palatino']})
           >>> rc('text', usetex=True)
            
           #parameters
            
           >>> stickout_length='2'
           >>> WPT_Level = 4
           >>> Classifier = 'SVC'
           >>> plotting = True
         
           >>> results = WPT_Feature_Extraction(stickout_length, WPT_Level, 
           >>>                                  Classifier, plotting)     
           Enter the path of the data files:
           >>> D\...\cutting_tests_processed\data_2inch_stickout
           
       .. image:: example.jpg
          :width: 600px
          :height: 360px
   
    """
    user_input = input("Enter the path of the data files: ")

    assert os.path.exists(user_input), "Specified file does not exist at, "+str(user_input)
    
    folderToLoad = os.path.join(user_input)
    
    #%% start timer
    start2 = time.time()

    # Loading time series and labels of the classification
    
    # import the list including the name of the time series of the chosen case
    file_name = 'time_series_name_'+stickout_length+'inch.txt'
    file_path = os.path.join(folderToLoad, file_name)
    f = open(file_path,'r',newline='\n')
    
    #save the time series name into a list
    namets = []
    for line in f:
        names = line.split("\r\n")
        namets.append(names[0])
        
    #import the classification labels
    label_file_name = stickout_length+'_inch_Labels_2Class.npy'
    file_path1 = os.path.join(folderToLoad, label_file_name)
    label = np.load(file_path1)
    
    #%% Upload the Decompositions and compute the feature from them----------------
    #name of datasets

    numberofcase = len(namets) 
    featuremat= np.zeros((numberofcase,10))
    
    #load datasets and compute features
    for i in range (0,numberofcase):
        nameofdata = 'WPT_Level%s_Recon_%sinch_%s' %(str(WPT_Level),stickout_length,namets[i])
        pathofdata = os.path.join(folderToLoad, nameofdata)
        ts = sio.loadmat(pathofdata)
        ts= ts["recon"]

        featuremat[i,0] = np.average(ts)
        featuremat[i,1] = np.std(ts)
        featuremat[i,2] = np.sqrt(np.mean(ts**2))   
        featuremat[i,3] = max(abs(ts))
        featuremat[i,4] = skew(ts)
        L=len(ts)
        featuremat[i,5] = sum(np.power(ts-featuremat[i,0],4)) / ((L-1)*np.power(featuremat[i,1],4))
        featuremat[i,6] = featuremat[i,3]/featuremat[i,2]
        featuremat[i,7] = featuremat[i,3]/np.power((np.average(np.sqrt(abs(ts)))),2)
        featuremat[i,8] = featuremat[i,2]/(np.average((abs(ts))))
        featuremat[i,9] = featuremat[i,3]/(np.average((abs(ts))))

    
    #%% load frequency domain features (At different levels of WPT) and combine them 
    #   with the time domain features
    freq_feature_file_name = 'WPT_Level%d_Freq_Features_%sinch.mat'%(WPT_Level,stickout_length)
    file_path_Ff = os.path.join(folderToLoad, freq_feature_file_name)        
    freq_features = sio.loadmat(file_path_Ff)
    freq_features = freq_features['Freq_Features']
    
    #concatanate the frequency and time domain features 
    featuremat = np.concatenate((featuremat, freq_features),axis = 1)
    n_feature=14
    #%% Classification
    
    #generating accuracy, meanscore and deviation matrices
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
        F_train,F_test,Label_train,Label_test= train_test_split(featuremat,label, test_size=0.33)
        
        F_traincomb = np.zeros((len(F_train),n_feature))
        F_testcomb = np.zeros((len(F_test),n_feature))
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
        Label_train=np.ravel(Label_train)
        selector = selector.fit(F_train, Label_train)
        rank = selector.ranking_
        Rank.append(rank)
        rank = np.asarray(rank)
        
        #create a list that contains index number of ranked features
        rankedlist = np.zeros((14,1))
    
        #finding index of the ranked features and creating new training and test sets with respect to this ranking
        for m in range (1,15):
            k=np.where(rank==m)
            rankedlist[m-1]=k[0][0]
            F_traincomb[:,m-1] = F_train[:,int(rankedlist[m-1][0])]
            F_testcomb[:,m-1] = F_test[:,int(rankedlist[m-1][0])] 
        RankedList.append(rankedlist)
        
        #trying various combinations of ranked features such as ([1],[1,2],[1,2,3]...)
        for p in range(0,14): 
            start1 = time.time()
            clf.fit(F_traincomb[:,0:p+1],Label_train)
            score1=clf.score(F_testcomb[:,0:p+1],Label_test)
            score2=clf.score(F_traincomb[:,0:p+1],Label_train)
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
            
        
    #%% plotting mean accuracies and deviations
    if plotting == True:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        line1 = ax1.plot(meanscore1,'b-',label=r'Test set score')
        line2 = ax1.plot(meanscore2,'g-',label=r'Training set score')
        line3 = ax2.plot(deviation1,'r--',label=r'Test set deviation')
        line4 = ax2.plot(deviation2,'c--',label=r'Training set deviation')
        lines = line1+line2+line3+line4
        labs = [l.get_label() for l in lines]
        Fontsize = 25
        ax1.legend(lines, labs, loc=1, fontsize = Fontsize )
        ax1.set_xlabel(r'Number of Features',fontsize = Fontsize)
        ax1.set_ylabel(r'Score of Classification',fontsize =Fontsize)
        ax2.set_ylabel(r'Deviation',fontsize = Fontsize)
        for tick in ax1.xaxis.get_major_ticks():
            tick.label.set_fontsize(Fontsize) 
        for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(Fontsize) 
        ax2.tick_params(labelsize = Fontsize)
        plt.show()
        #plt.savefig('Number_of_features_vs_deviation_accuracy.pdf',bbox_inches = 'tight', dpi=300)
        
    return results,print('Total elapsed time: {}'.format(duration2)),featuremat
