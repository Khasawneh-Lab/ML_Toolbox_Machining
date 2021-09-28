import time
import numpy as np
import scipy.io as sio
import pywt

def WPT_Informative_Packet_Recon(data_path,list_name,WF,L,IWP,saving, *args):
    
    '''
    This function reconstructs signals using the coeffients of informative wavelet packets.
    
    :param str (data_path): Path to folder where list of time series names and time series data is kept
    
    :param str (list_name): Name of the .txt file which includes names of time series data
    
    :param list (label_name): Name of the .npy file which includes labels of time series
    
    :param str (WF): Wavelet function (see types_ of wavelet functions available in PyWavelet package)

    :param int (IWP): Informative Wavelet Packet Number (IWP)

    :param int (L): Transformation level 
    
    :param bool (saving): It is set to 'True' to save the reconstructed signals
    
    :param list (*args): Additional parameters such as saving file name for the reconstructed signals if saving is set to true.
            
    :Returns:
        :recon:
            Object array that contains the reconstructed signals using informative wavelet packet defined by user.
        :time:
            Time elapsed during the reconstruction of the signals
    :Example:            
        .. doctest::    
            
        from WPT_Informative_Packet_Recon import WPT_Informative_Packet_Recon
        
        # parameters
        
        data_path = 'D:\\Data Archive\\Cutting_Test_Data_Documented\\cutting_tests_processed\\2inch_stickout'
        list_name = 'time_series_name_2inch.txt'
        WF = 'db10'
        L=4
        IWP = 3
        saving = False
    
        recon = WPT_Informative_Packet_Recon(data_path,list_name,WF,L,IWP,saving)        
    
    '''
    
    start=time.time()
    # file path to folder where the data is kept
    file_path = data_path+'\\'+list_name
    
    # read the file that includes the name of datafiles
    with open(file_path) as f:
        data_names = f.read().splitlines()
        
    N = len(data_names)
    
    if IWP<1 or IWP>2**L:
        raise Exception('Invalid informative wavelet packet number!')
    

    ts = np.zeros((N),dtype=object)
    recon = np.zeros((N),dtype=object)
    
    # compute wavelet coefficients and energy ratios
    for i in range(N):
        # load mat files
        name = data_names[i]

        #load the time series
        ts[i] = sio.loadmat(data_path+'\\'+name)['tsDS']    
    
    
        # apply wavelet packet decomposition
        wp = pywt.WaveletPacket(data=ts[i][:,1], wavelet=WF,maxlevel=L,mode='symmetric')
        packet_order = [node.path for node in wp.get_level(L, 'freq')]
        
        # reconstruct the wavelet packets
        # create a empty wavelet packet object
        new_wp = pywt.WaveletPacket(data=None, wavelet=WF, mode='symmetric')
        new_wp[packet_order[IWP-1]] = wp[packet_order[IWP-1]]
        recon[i] = new_wp.reconstruct(update=False)
        # find the diff between reconstructed length and time series length
        if len(ts[i])>len(recon[i]):
            index = len(recon[i])
        else:
            index = len(ts[i])
        recon[i] = recon[i][:index]
        
        if saving:
            np.save(data_path+'\\'+args[0],recon)

    end = time.time()
    print("Elapsed time for reconstruction: {} seconds".format(end-start))
                  
    return recon