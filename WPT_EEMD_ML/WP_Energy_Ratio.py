
import numpy as np
import pywt
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def wEnergy(coeffs):
    
    '''
    This function takes wavelet coefficients and return the energy ratio of wavelet packets.
    
    :param list (coeffs): The list of wavelet coefficients obtained from Wavelet Packet Transform
    
    :Returns:
        (np.array([])) The energry ratios of each wavelet packets 
    
    
    '''
  
    Et = 0
    energy = np.zeros(len(coeffs))
    for k in range(len(coeffs)):
        energy[k] = sum(coeffs[k]**2)
        Et = Et + sum(coeffs[k]**2)
        
    en_ratio = energy/Et
    
    return en_ratio

def AbsFFT(ts, fs):

    '''
    This function computes the normaliez spectrum of the given 1D signal.
    
    :param array (ts): time series
    
    :param int (fs): sampling frequency
    
    :Returns:
        :xf:
            (np.array([])) frequency values
        
        :yf:
            (np.array([])) amplitudes
          
    '''
    fs = fs/2
    ts = ts.reshape(len(ts,))
    t = 1/fs 
    N = len(ts)
    
    #creates array for time data points to associate to time series
    xf = np.split(np.linspace(0.0, 1.0/t, N//2),[1])[1] 
    #converts array of time data points to appropriate frequency range and removes first term
    
    yf = fft(ts) #computes fast fourier transform on time series
    yf = (np.abs(yf[0:N//2])) #converts fourier transform to spectral density scale
    yf = np.split(yf,[1])[1] #removes first term (tends to be infinity or overly large)
    yf = yf/max(yf) #Normalizes fourier transform based on maximum density
    return(xf, yf)

def WP_Energy_Ratio(data_path, list_name, ts_no, WF, L, case_no, fs, plot_recon, layout=None ):
    
    '''
    This function returns plots of energy ratios of wavelet packets, reconstructed signals of each wavelet packet and their spectrum.
    Reconstructed signals is plotted when plot_recon is set to 'True'.
    
    :param str (data_path): Path to folder where list of time series names and time series data is kept
    
    :param str (list_name): Name of the .txt file which includes names of time series data
    
    :param list (ts_no): Index of time series which user wants to investigate
    
    :param str (WF): Wavelet function (see types_ of wavelet functions available in PyWavelet package)

    :param int (L): Transformation level 
    
    :param int (case_no): Overhang distance 
    
    :param int (fs): Sampling frequency
    
    :param bool (plot_recon): It is set to 'True' to obtain the reconstructed signals plots 
    
    :param list (layout): Row and column numbers for the plot of reconstructed signals 
    
    .. _types: https://pywavelets.readthedocs.io/en/latest/regression/wavelet.html?highlight=wavelet%20list#wavelet-families-and-builtin-wavelets-names
        
    :Returns:
        :figures:
            First figure is the energy ratio plot. If user set to plot_recon to 'True', then algorithm returns reconstruted signals figures as well.
            For instance, if the number of time series that user wants to investigate is 3, and algorithm returns two figures for each time series.
            Totally, it returns 7 figures. 
            Plots of reconstructed signals and their spectrums are given in the order user specified as input with ts_no.
        
          
    '''
    
    # font size for the labels in plots
    f_s = 15
    
    # file path to folder where the data is kept
    file_path = data_path+'\\'+list_name
    
    # number of time series which will be investigated 
    N = len(ts_no)
    
    # read the file that includes the name of datafiles
    with open(file_path) as f:
        data_names = f.read().splitlines() 
      
    rpm = []
    doc = []
    label = []
    legend_name=[]
    ts = np.zeros((N),dtype=object)
    ratio = np.zeros((N,2**L))
    recon = np.zeros((N,2**L),dtype=object)
    
    # compute wavelet coefficients and energy ratios
    for i in range(N):
        # load mat files
        name = data_names[ts_no[i]-1]
        
        # time series information
        rpm.append(str(name[2:5]))
        doc.append(str('0.'+name[6:9]))
        
        if name[0]=='c':
            label.append('Chatter')
        elif name[0]=='s':
            label.append('Stable')
        elif name[0]=='i':
            label.append('Mild Chatter')
            
        legend_name.append(label[i]+', Stickout='+str(case_no)+' inch, RPM='+str(name[2:5])+', DOC='+str('0.'+name[6:9]))
        
        #load the time series
        ts[i] = sio.loadmat(data_path+'\\'+name)['tsDS']
        
        # apply wavelet packet decomposition
        wp = pywt.WaveletPacket(data=ts[i][:,1], wavelet=WF,maxlevel=L,mode='symmetric')
        packet_order = [node.path for node in wp.get_level(L, 'freq')]
        coeffs = [wp[path].data for path in packet_order]
        
        #reconstruct the wavelet packets
        for j in range(2**L):
            # create a empty wavelet packet object
            new_wp = pywt.WaveletPacket(data=None, wavelet=WF, mode='symmetric')
            new_wp[packet_order[j]] = wp[packet_order[j]]
            recon[i,j] = new_wp.reconstruct(update=False)
            # find the diff between reconstructed length and time series length
            len_dif = len(ts[i])-len(recon[i,j])
            recon[i,j] = recon[i,j][:len_dif]
            
        #energy ratio
        ratio[i,:] = wEnergy(coeffs)
        
        
    # plot energy ratios   
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 5)) 
       
    for i in range(N):
        # Set position of bar on X axis
        if i == 0:
            br = np.arange(2**L)+1
        else:
            br = [x + barWidth for x in br]
            
        plt.bar(br, ratio[i,:], width = barWidth, label = legend_name[i])
    
    plt.xlabel('Wavelet Packets' , fontsize = f_s)
    plt.ylabel('Energy Ratios', fontsize = f_s)    
    plt.legend()  
    plt.subplots_adjust(hspace=0.5,wspace=0.5)
    ax=plt.gca()
    ax.tick_params(axis='both', labelsize=f_s-1)
    
    # plot reconstructed signals and their spectrum
    if plot_recon:
        for i in range(N):
            fig, axes = plt.subplots(nrows=layout[0], ncols=layout[1],figsize=(12,9)) 
            fig_FFT, axes_FFT = plt.subplots(nrows=layout[0], ncols=layout[1],figsize=(12,9))
             
            axes = axes.flatten()
            axes_FFT = axes_FFT.flatten()
    
            
            for j in range(2**L):   
                ax = axes[j]
                ax.plot(ts[i][:,0],recon[i,j])
                ax.set_xlabel('time (seconds)' , fontsize = f_s)
                ax.set_ylabel('Amplitude', fontsize = f_s)  
                title = 'Packet'+str(j+1)
                ax.set_title(title)
                ax.tick_params(axis='both', labelsize=f_s-1)
                
                plt.subplots_adjust(hspace=1 ,wspace=1)
                
                ax = axes_FFT[j]
                xf, yf = AbsFFT(recon[i,j], fs)
                ax.plot(xf,yf)
                ax.set_xlabel('frequency (Hz)' , fontsize = f_s)
                ax.set_ylabel('$\mid X(f) \mid $', fontsize = f_s)  
                title = 'Packet'+str(j+1)
                ax.set_title(title, fontsize = f_s)
                ax.tick_params(axis='both', labelsize=f_s-1)
                
                plt.subplots_adjust(hspace=0.8,wspace=0.5)
                