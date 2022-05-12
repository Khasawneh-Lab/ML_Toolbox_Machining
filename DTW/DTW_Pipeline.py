"""
    This library contains the sources codes that enables users to classify time series 
    data based on the similarity measure technique called Dynamic Time Warping. 
    Chatter detection application was performed using this approache in Reference :cite:`Yesilli2022`. 
"""

import numpy as np
from dtw import *
import sys,os
from itertools import combinations
import multiprocessing
from multiprocessing import Pool
import time
from scipy.spatial.distance import squareform
from sklearn.model_selection import train_test_split


def DTW_Distance_Comp(ts1,ts2,WS,path_plot,dist_only):
    """
    

    Parameters
    ----------
    ts1 : np.array
        First time series
    ts2 : np.array
        Second time series
    WS : int
        Window size for sakoeChibaWindow
    path_plot : boolean
        Set it to True if you would like to see the warping path plot
    dist_only : boolean
        Set it to True if you woyld like to only compute the distance for faster computation
    Returns
    -------
    int
        DTW distance between given two time series

    """
    DTW = dtw(ts1, ts2,keep_internals=True,window_type=sakoeChibaWindow,window_args={'window_size': WS} , dist_method='cityblock',step_pattern='symmetricP0',distance_only=dist_only)
    
    if path_plot:
        DTW.plot(type="threeway")
        
    return DTW.distance

def DTW_Dist_Mat(TS1,TF,parallel,*args):
    """
    

    Parameters
    ----------
    TS1 : list
        The list of time series
    TF : boolean
        Set it to True if transfer learning will be applied
    parallel : boolean
        Set it to True for parallel computing of distance matrix
    *args : 
        TS2 : list 
            The list that includes the test set time series. This input is given only when
            TF is True.

    Returns
    -------
    DM : 2D np.array
        Distance matrix

    """
    # if user wants to apply transfer learning
    if TF:
        TS2 = args[0]
        
        if not parallel:
            # test set distance matrix-----------
            
            # length of combinations
            start = time.time()
            DM = np.zeros((len(TS1),len(TS2)))
            for i in range(len(TS1)):
                for j in range(len(TS2)):
                    x=np.ravel(TS1[i])
                    y=np.ravel(TS2[j])
                    DM[i,j]=DTW_Distance_Comp(x,y,1000,False,True)
            end = time.time()
            print("Elapsed time:{}".format(end-start))
        else:
            inputs = []
            for i in range(len(TS1)):
                for j in range(len(TS2)):
                    x=np.ravel(TS1[i])
                    y=np.ravel(TS2[j])
                    inputs.append((x,y,1000,False,True))
            
            # time the paralell computation of distances
            start = time.time()
            n_proc = multiprocessing.cpu_count()
            with Pool(processes=n_proc//2) as p:
                DM_cp  = p.starmap(DTW_Distance_Comp, inputs)            
            end = time.time()
            print("Elapsed time:{}".format(end-start)) 
            DM_cp = np.asarray(DM_cp)
            DM = DM_cp.reshape((len(TS1),len(TS2)))
            
        return DM
        
        # training set distance matrix
        
    else:
        # compute the combinations between each time series
        poss_comb = np.array(list(combinations(range(0,len(TS1)), 2)))
        # lenght of combinations
        N = len(poss_comb)
        # the loop that computes the distances for all combinations
        DM_cp=[]
        if not parallel:
            start = time.time()
            for i in range(N):
                x=np.ravel(TS1[poss_comb[i,0]])
                y=np.ravel(TS1[poss_comb[i,1]])
                DM_cp.append(DTW_Distance_Comp(x,y,1000,False,True))
            end = time.time()
            print("Elapsed time:{}".format(end-start))
            DM = squareform(DM_cp)
        else:  
            inputs = []
            for i in range(N):
                x=np.ravel(TS1[poss_comb[i,0]])
                y=np.ravel(TS1[poss_comb[i,1]])
                inputs.append((x,y,1000,False,True))
            
            # time the paralell computation of distances
            start = time.time()
            n_proc = multiprocessing.cpu_count()
            with Pool(processes=n_proc//2) as p:
                DM_cp  = p.starmap(DTW_Distance_Comp, inputs)            
            end = time.time()
            print("Elapsed time:{}".format(end-start))
            DM = squareform(DM_cp)
            
    return DM

def kNN(k,L,b,Label_train,labels):
    """
    

    Parameters
    ----------
    k : int
        Nearest neighbor number
    L : int
        the number of time series in the training or test set
    b : 2D np.array
        the distance matrix between training or test set samples
    Label_train : np.array
        The array that incldues the training set labels
    labels : np.array
        the array that incldues the test set labels if user wants to compute test set accuray or 
        it is training set samples for computing training set accuracy

    Returns
    -------
    true : int
        number of true predictions
    false : int
        number of false predictions

    """
    true = 0
    false = 0
    
    #test set classification
    
    for i in range(0,L):
        if k==1:
            distances = b[:,i]
            min_dist = min(distances)
            index = np.where(distances == min_dist)[0][0]
            pred = Label_train[index]
            test_class = labels[i]
            if test_class == pred:
                true = true+1
            else:
                false = false + 1
        if k>1:
            distances = b[:,i]
            Sort_dist = sorted(distances)
            del Sort_dist[k:]
            index = np.zeros((k))
            for m in range (0,k):
                index[m]=np.where(distances == Sort_dist[m])[0]
            index = index.astype(int)
            pred_neighbour = Label_train[index]
            pred_neighbour = pred_neighbour.tolist()
            numberofzeros = pred_neighbour.count(0)
            numberofones = pred_neighbour.count(1)
            if numberofzeros > numberofones:
                pred = 0
            else:
                pred = 1
            test_class = labels[i]
            if test_class == pred:
                true = true+1
            else:
                false = false + 1
    
    return true,false
                    
def TS_Classification(k,TS1,labels1,DM1,TF,*args):
    """
    

    Parameters
    ----------
    k : int
        The nearest neighbor number.
    TS1 : list
        The list that includes the time series
    labels1 : np.array
        The list of labels belong to the set of time series
    DM1 : 2D np.array
        The pairwise distance matrix between the time series
    TF : boolean
        Set tot True if transfer learning will be applied
    *args : 
        TS2 : list
            The list that includes the time series in the test set  when transfer learning is selected True
        labels2 : np.array
            The list of labels for the test set
        DM2 : 2D np.array
            The distance matrix for the test set time series data
        
    Returns
    -------
    output : dict
        Includes mean accuracy and standard deviation for training and test set.

    """
    accuracy_test =np.zeros((10))
    accuracy_train =np.zeros((10))    
    if not TF:
        start_time_classification = time.time()
        for T in range(10):
            #split time series into test set and training set
            TS_train,TS_test, Label_train, Label_test = train_test_split(TS1, labels1, test_size=0.33)
            
            TS_train_L = len(TS_train)
            TS_test_L = len(TS_test)
             
            Label_train = Label_train
            Label_test = Label_test
            
            #find training set indices
            TS_train_index = np.zeros((TS_train_L))
            for i in range(0,TS_train_L):
                train_sample = TS_train[i]
                for j in range(0,len(TS1)):
                    if np.array_equal(TS1[j],train_sample):
                        TS_train_index[i] = j 
            #find test set indices
            TS_test_index = np.zeros((TS_test_L))
            for i in range(0,TS_test_L):
                test_sample = TS_test[i]
                for j in range(0,len(TS1)):
                    if np.array_equal(TS1[j],test_sample):
                        TS_test_index[i] = j             
            
            #generate distance matrix between test set and training set
            b = np.zeros(shape=(TS_train_L,TS_test_L))
            for i in range (0,TS_train_L):
                train_index = int(TS_train_index[i])
                for j in range(0,TS_test_L):
                    test_index = int(TS_test_index[j])
                    b[i,j]=DM1[train_index,test_index]
                    
            #k-NN classification
            true,false = kNN(k,TS_test_L,b,Label_train,Label_test)
            accuracy_test[T] = true/(true+false)
            
            # the distance matrix between the training set samples
            b2 = np.zeros(shape=(TS_train_L,TS_train_L))
            for i in range (0,TS_train_L):
                train_index_1 = int(TS_train_index[i])
                for j in range(0,TS_train_L):
                    train_index_2 = int(TS_train_index[j])
                    b2[i,j]=DM1[train_index_1,train_index_2]
            
            #training set classification
            true,false = kNN(k,TS_train_L,b2,Label_train,Label_train)
            accuracy_train[T] = true/(true+false)
            
        
        mean_score_test = np.mean(accuracy_test)
        deviation_test=np.std(accuracy_test) 
        mean_score_train = np.mean(accuracy_train)
        deviation_train = np.std(accuracy_train)
        end_time_classification = time.time()
        classfication_time = end_time_classification - start_time_classification
    
        print('Score(Test Set): {} Deviation(Test Set): {}'.format(mean_score_test,deviation_test))
        print('Score(Training Set): {} Deviation(Training Set): {}'.format(mean_score_train,deviation_train))
        print('Classification Time: {}'.format(classfication_time))
        
        output = {}
        output['mean_test_score'] = mean_score_test
        output['dev_test']= deviation_test
        output['mean_train_score'] = mean_score_train
        output['dev_train'] = deviation_train
        output['clf_time'] = classfication_time
    else:
        TS2 = args[0]
        labels2 = args[1]
        DM2 = args[2]
        start_time_classification = time.time()
        for T in range(0,10):
            #split time series into test set and training set
            TS_training_train,TS_training_test, Label_train_train, Label_train_test = train_test_split(TS1, labels1, test_size=0.33)
            TS_test_train,TS_test_test, Label_test_train, Label_test_test = train_test_split(TS2, labels2, test_size=0.67)
            
            TS_train_L = len(TS_training_train)
            TS_test_L = len(TS_test_test)
             
            Label_train = Label_train_train
            Label_test = Label_test_test
            
            #find training set indices
            TS_train_index = np.zeros((TS_train_L))
            for i in range(0,TS_train_L):
                train_sample = TS_training_train[i]
                for j in range(0,len(TS1)):
                    if np.array_equal(TS1[j],train_sample):
                        TS_train_index[i] = j 
            #find test set indices
            TS_test_index = np.zeros((TS_test_L))
            for i in range(0,TS_test_L):
                test_sample = TS_test_test[i]
                for j in range(0,len(TS2)):
                    if np.array_equal(TS2[j],test_sample):
                        TS_test_index[i] = j             
            
            #generate distance matrix between test set and training set
            b = np.zeros(shape=(TS_train_L,TS_test_L))
            for i in range (0,TS_train_L):
                train_index = int(TS_train_index[i])
                for j in range(0,TS_test_L):
                    test_index = int(TS_test_index[j])
                    b[i,j]=DM1[train_index,test_index]
                    
            #test set classification
            true,false = kNN(k,TS_test_L,b,Label_train,Label_test)                            
            accuracy_test[T] = true/(true+false)

            #distance matrix for training set
            b2 = np.zeros(shape=(TS_train_L,TS_train_L))
            for i in range (0,TS_train_L):
                train_index_1 = int(TS_train_index[i])
                for j in range(0,TS_train_L):
                    train_index_2 = int(TS_train_index[j])
                    b2[i,j]=DM2[train_index_1,train_index_2]
            
            #training set classification
            true,false = kNN(k,TS_train_L,b2,Label_train,Label_train)
            accuracy_train[T] = true/(true+false)

        mean_score_test = np.mean(accuracy_test)
        deviation_test=np.std(accuracy_test) 
        mean_score_train = np.mean(accuracy_train)
        deviation_train = np.std(accuracy_train)
        end_time_classification = time.time()
        classfication_time = end_time_classification - start_time_classification
        
        print('Score(Test Set): {} Deviation(Test Set): {}'.format(mean_score_test,deviation_test))
        print('Score(Training Set): {} Deviation(Training Set): {}'.format(mean_score_train,deviation_train))
        print('Classification Time: {}'.format(classfication_time))
        
        output = {}
        output['mean_test_score'] = mean_score_test
        output['dev_test']= deviation_test
        output['mean_train_score'] = mean_score_train
        output['dev_train'] = deviation_train
        output['clf_time'] = classfication_time
                 
        end_time_classification = time.time()
        classfication_time = end_time_classification - start_time_classification        
    
    return output



if __name__ == '__main__':
    # generate two time series
    ts1 = np.linspace(0,6.28,num=100)
    ts2 = np.sin(ts1) + np.random.uniform(size=100)/10.0
    
    # compute the distance between and the plot the warping path
    distance = DTW_Distance_Comp(ts1,ts2,5,True,False)
    
    # generate synthetic data set
    
    TS1 = []
    for i in range(15): 
        fs, T = 100, 10 
        t = np.linspace(-0.2,T,fs*T+1) 
        A = 20 
        TS1.append(A*np.sin((i+1)*np.pi*t) + A*np.sin(1*t))
    
    # serial distance computation
    DM1 = DTW_Dist_Mat(TS1,False,False)
    # parallel distance computation
    DM2 = DTW_Dist_Mat(TS1,False,True)
    
    # Generate the second set of time series so that we can apply transfer learning
    TS2 = []
    for i in range(20): 
        fs, T = 100, 10 
        t = np.linspace(-0.2,T,fs*T+1) 
        A = 20 
        TS2.append(A*np.sin((2*i+1)*np.pi*t) + A*np.sin(2*t))   
    
    # serial distance computation
    DM_TF1 = DTW_Dist_Mat(TS1,True,False,TS2)
    # parallel distance computation
    DM_TF2 = DTW_Dist_Mat(TS1,True,True,TS2)    
    
    # perform classification
    labels1 = np.random.choice([0, 1], size=(len(TS1),), p=[1./3, 2./3])
    out = TS_Classification(1,TS1,labels1,DM1,False)
    
    # perform classification using transfer learning
    labels2 = np.random.choice([0, 1], size=(len(TS2),), p=[1./3, 2./3])
    out = TS_Classification(1,TS1,labels1,DM_TF1,True,TS2,labels2,DM_TF2)
    
    