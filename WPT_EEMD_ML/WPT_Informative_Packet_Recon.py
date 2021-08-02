
import numpy as np
import scipy.io as sio
import pywt


# inputs 
data_path = 'D:\\Data Archive\\Cutting_Test_Data_Documented\\cutting_tests_processed\\2inch_stickout'
list_name = 'time_series_name_2inch.txt'
label_name = '2_inch_Labels_2Class.npy'
WF = 'db10'
L=4
IWP = 3
save_name = 'Recon_test'
saving = True


def WPT_Informative_Packet_Recon(data_path,list_name,label_name,WF,L,IWP,saving, *args):
    
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
          
    '''
    # file path to folder where the data is kept
    file_path = data_path+'\\'+list_name
    label_path = data_path+'\\'+label_name
    label = np.load(label_path)
    
    
    # read the file that includes the name of datafiles
    with open(file_path) as f:
        data_names = f.read().splitlines()
        
    N = len(data_names)
    
    if IWP<1 or IWP>2**L:
        raise Exception('Invalid informative wavelet packet number!')
    
    
    rpm = []
    doc = [] 
    ts = np.zeros((N),dtype=object)
    recon = np.zeros((N,1),dtype=object)
    
    # compute wavelet coefficients and energy ratios
    for i in range(N):
        # load mat files
        name = data_names[i]
        
        # time series information
        rpm.append(str(name[2:5]))
        doc.append(str('0.'+name[6:9]))
        
        #load the time series
        ts[i] = sio.loadmat(data_path+'\\'+name)['tsDS']    
    
    
        # apply wavelet packet decomposition
        wp = pywt.WaveletPacket(data=ts[i][:,1], wavelet=WF,maxlevel=L,mode='symmetric')
        packet_order = [node.path for node in wp.get_level(L, 'freq')]
        coeffs = [wp[path].data for path in packet_order]
        
        # reconstruct the wavelet packets
        # create a empty wavelet packet object
        new_wp = pywt.WaveletPacket(data=None, wavelet=WF, mode='symmetric')
        new_wp[packet_order[IWP-1]] = wp[packet_order[IWP-1]]
        recon[i,0] = new_wp.reconstruct(update=False)
        # find the diff between reconstructed length and time series length
        len_dif = len(ts[i])-len(recon[i,0])
        recon[i,0] = recon[i,0][:len_dif]
        
        if saving:
            np.save(save_name,recon)
        return recon