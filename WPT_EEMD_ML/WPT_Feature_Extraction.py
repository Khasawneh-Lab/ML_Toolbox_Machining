import time
import numpy as np
from scipy.stats import skew
from WPT_EEMD_ML.WPT_Informative_Packet_Recon import WPT_Informative_Packet_Recon
from WPT_EEMD_ML.WP_Energy_Ratio import AbsFFT


def WPT_Feature_Extraction(data_path, list_name,label_name,WF,L,IWP,fs,saving,*args):
    """
    
    :param str (data_path): The path where user keeps the data set
  
    :param str (list_name): Name of the .txt file that includes the names of time series data 
    
    :param str (label_name): Name of the .npy file that includes labels of the time series
        
    :param str (WF): Wavelet function
       	
    :param  int (L): Level of the transform which will be applied to data set
    
    :param int (IWP): Informative Wavelet Packet number
        
    :param int (fs): Sampling frequency of the data set
    
    :param boolean (saving): Set it to true if you want to save reconstructed signals 
    
    :Returns:

        :feature_mat:
            (np.array([])) Feature matrix 
           
        :labels: 
            (np.array([])) Labels
            
        :time:
            (str) Elapsed time during feature matrix generation 
    
    :Example:
    
        .. doctest::
                       
           from WPT_EEMD_ML.WPT_Feature_Extraction import WPT_Feature_Extraction
           
           # parameters
            
           data_path = 'D:\\Data Archive\\Cutting_Test_Data_Documented\\cutting_tests_processed\\2inch_stickout'
           list_name = 'time_series_name_2inch.txt'
           WF = 'db10'
           L=4
           IWP = 3
           label_name = '2_inch_Labels_2Class.npy'
           saving = False
           fs = 10000 
         
           feature_mat,labels = WPT_Feature_Extraction(data_path, list_name,label_name,WF,L,IWP,fs,saving)
   
    """

    #%% Loading time series and labels
    start = time.time()
    
    # file path to folder where the data is kept
    file_path = data_path+'\\'+list_name
    label_path =  data_path+'\\'+label_name

    # read the file that includes the name of datafiles
    with open(file_path) as f:
        data_names = f.read().splitlines()
        
    N = len(data_names)
   
    # import the classification labels
    label = np.load(label_path)

    # reconstruct signals from informative wavelet packet
    if saving:
        recon = WPT_Informative_Packet_Recon(data_path,list_name,WF,L,IWP,saving,args[0])
    else:
        recon = WPT_Informative_Packet_Recon(data_path,list_name,WF,L,IWP,saving)   
        
    # compute features
    featuremat= np.zeros((N ,14))
        
    for i in range(N):

        ts = recon[i]
        
        # compute time domain features
        
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

        # compute FFT of the reconstructed signals
        xf,yf = AbsFFT(recon[i],fs)
        
        # frequency domain features
        featuremat[i,10] = sum((xf**2)*yf)/sum(yf) #mean square frequency
        featuremat[i,11] = sum(np.cos(2*np.pi*xf*1/fs)*yf)/sum(yf) # one step auto correlation function
        featuremat[i,12] = sum(xf*yf)/sum(yf) # frequency center
        featuremat[i,13] = sum(((xf-featuremat[i,12])**2)*yf)/sum(yf) # standard frequency
        
        

    # concatanate feature matrix and the label matrix and then shuffle them
    feat_lab = np.concatenate((featuremat,np.reshape(label,(len(label),1))),axis=1)
    np.random.shuffle(feat_lab)
    
    feature_mat = feat_lab[:,0:14]
    labels = feat_lab[:,14]
    end = time.time()

    if saving:
        output={}
        output['features'] = feature_mat
        output['labels'] = labels
        np.save(data_path+'\\'+args[0],output)    

    print("Feature computation is completed in {} seconds.".format(end-start))
    
    return feature_mat,labels
    
  