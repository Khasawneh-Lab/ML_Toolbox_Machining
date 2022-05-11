"""
Feature extraction using Ensemble Empirical Mode Decomposition 
--------------------------------------------------------------



"""
import time
start2 = time.time()
import numpy as np
from PyEMD import EEMD
import scipy.io as sio
import os.path
import sys
from scipy.stats import skew


def preprocess_chatter_data(data_path,list_name):
    """
    This function is specifally designed for the chatter data set where we have 
    long time series whose IMF computation takes longer. 
    This function split the time series into small pieces to reduce the computation time. 

    Parameters
    ----------
    data_path : str
        The path to data files.
    list_name : str
        The name of the txt file that contains the list of names of time series. 

    Returns
    -------
    split_data : Object Array
        Object array that includes the split data 
    split_labels : ndarray
        2D array which contains the labels of the split data

    """
    # file path to folder where the data is kept
    file_path = data_path+'\\'+list_name

    # read the file that includes the name of datafiles
    with open(file_path) as f:
        data_names = f.read().splitlines()
        
    N = len(data_names)
   
    # Loading time series and labels of the classification
    
    # import the time series
    N = len(data_names)     
    data_L = []
    data = {}
    #load datasets and compute features
    for i in range (0,N):
        name = '%s' %(data_names[i])
        ts = sio.loadmat(data_path+'\\'+name)['tsDS'] 
        L = len(ts)
        data_L.append(L)
        
        # extract rpm and depth of cut information from name of the data set
        rpm = np.full((L,1),int(name[2:5]))
        if name[5]=='_':
            doc = np.full((L,1),int(name[6:9])*0.001)
        else:
            doc = np.full((L,1),int(name[7:10])*0.001)
            
        # time series with "i" and "c" represents 
        if name[0] == 's':
            stability = np.full((L,1),0)
        else:
            stability = np.full((L,1),1)
        
        data[i] = np.concatenate((ts,rpm,doc,stability),axis=1)
    
    # total data points in whole data set
    total_L = sum(data_L)
    
    # find the approximate number of time series after we split them into small ones
    app_num_ts = total_L//1000
    
    # generate arrays for the split data
    split_data = np.ndarray(shape=(app_num_ts),dtype=object)
    split_labels = np.ndarray(shape=(app_num_ts,3))
    inc = 0 # increment that counts the number of total splits
    for i in range(N):
        ts = data[i]
    
        # splitting 
        if len(ts)>1000:
            num_splits = len(ts)//1000
            split = np.array_split(ts,num_splits) 
            
            for j in range(num_splits):
                split_data[inc] = split[j][:,0:2]
                split_labels[inc] = split[j][0,2:5]
                inc = inc+1
    # remove the empty rows from the split arrays
    split_data = split_data[:inc]
    split_labels = split_labels[:inc]
    
    return split_data, split_labels

def EEMD_IMF_Compute(data_path,list_name, EEMDecs, saving, *args):
    """
    This function takes set of time series and compute their IMFs. If time series
    data contains large samples, it uses the preprocessing function to reduce the 
    computation time. Resulting split time series is used to compute IMFs

    Parameters
    ----------
    data_path : str
        The path to data files.
    list_name : str
        The name of the txt file that contains the list of names of time series. 
    EEMDecs : str
        The parameter that defines if the IMFs are precomputed or not. It is set to 
        "A" if the IMFs for the current data set is available. It is set to "NA", if IMFs 
        are not computed yet.
    saving : Boolean
        If user wants to save the decomposition for the data set, this parameter  is set to 
        True. 
    *args : 
        saving_path: str
            The path to save the decomposition. This parameter is only needed when saving is 
            set to True.

    Returns
    -------
    infoEMF : object array
        Object array that includes decomposition for each split time series.
    split_labels : ndarray
        2D array that includes the labels of the split time series.

    """

    # split the data into small chunks to reduce time to compute IMFs
    split_data, split_labels = preprocess_chatter_data(data_path,list_name)
 
      
    # Compute IMFs if they are not computed before
    if EEMDecs=='NA':
        # generate the array that stores the decomposition
        infoEMF=np.ndarray(shape=(len(split_data)),dtype=object) 
        
        eemd = EEMD()
        emd = eemd.EMD
        emd.trials = 200      #default = 100
        emd.noise_width = 0.2 #default = 0.05

        # Chosen imf for feature extraction
        for i in range(0,len(split_data)): 
        
            #signal
            S = split_data[i][:,1]
            t = split_data[i][:,0]
            
            eIMFs = emd(S, t)
            infoEMF[i]=eIMFs  
            print('Progress: IMFs were computed for case number {}.  '.format(i))
    
        #save eIMFs into mat file
        if saving:
            saving_path = args[0]
            name = saving_path+'\\IMFs_Case_%i'%(i+1)
            np.save(name, infoEMF)
           
    elif EEMDecs=='A':
        load_path = args[0]
        dataname = load_path+'\\IMFs.npy'
        infoEMF = np.load(dataname,allow_pickle=True)
            
    return infoEMF,split_labels
    
def EEMD_Feature_Compute(infoEMF,p):
    """
    This function computes the features for selected IMF of each time series.

    Parameters
    ----------
    infoEMF : object array
        Object array that includes the IMFs for each time series.
    p : int
        The informative IMF number.

    Returns
    -------
    features : ndarray
        Feature matrix

    """
    features = np.zeros((len(infoEMF),7))
    
    for i in range(0,len(infoEMF)):
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
    
    return features
     
