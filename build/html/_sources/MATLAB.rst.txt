.. _MATLAB:

=========================================
MATLAB Codes for Wavelet Packet Transform
=========================================

MATLAB folder in the `GitHub repository <https://github.com/mcanyesilli/WPT_EEMD_ML_Machining>`_ includes four codes.
	
	* **WP_Energy_Ratio**: Computes the energy ratios of the wavelet packets for specified level of transform for chosen time series. It returns plots which contains energy ratios of each packet for time series, reconstructed time series and their FFT plot. User can define the informative wavelet packet number using these plot. 
	* **WP_Informative_Packet_Recon**: For selected level of WPT, it reconstructs time series based on chosen informative wavelet packets. It saves the reconstructed time series into specified folder.
	* **WP_Frequency_Domain_Features**: Computes the frequency domain features for selected level of Transform by using reconstructed time series with **WP_Informative_Packet_Recon** function. It saves the feature matrix and this feature matrix will be combined with time domain features to be used in classification.
	* **Example_Usage**: Provides examples for MATLAB codes.

**NOTE**: For more details on these functions, please see the function descriptions in MATLAB files.

Wavelet packet numbering in MATLAB
----------------------------------

The numbering of the wavelet packets obtained after applying WPT can confuse user. Therefore, we give proper instructions here how to obtain and track the wavelet packets correctly. There
are two types of ordering of the wavelet packets and these are natural ordering and frequency
ordering. We show these two types of ordering in Wavelet packet tree in :numref:`PacketNum`.	

.. _PacketNum:

.. figure:: WPT_Tree_with_Ordering.png
   :align: center

   Frequency and natural ordering of wavelet packets (resetting the packet numbers to 1 in each level of transformation)

The ordering is given in natural ordering in MATLAB. However, we use frequency ordering in this toolbox. In addition, MATLAB does not reset the numbering of wavelet packets to 1 in each level of transform. 
For example, if the first level wavelet packets are called first and second wavelet packets, first packet of the second level transform is called third wavelet packet. 
Please see :numref:`PacketNum_MATLAB` for MATLAB numbering.
The frequency ordering given in :numref:`PacketNum` is resetting the numbers to 1 in each level. 
All wavelet packet numbers given in :cite:`1 <Yesilli2019>` is based on the ordering provided in :numref:`PacketNum`.
One need to find correponding number for each wavelet packet by using :numref:`PacketNum_MATLAB` to reconstruct the time series correctly based on informative wavelet packets in MATLAB.
Also, ordering in :numref:`PacketNum` is obtained by using the MATLAB function **otnodes()**. 
Ordering can differ based on wavelet function used in the transform and level of the transform. 
Therefore, ordering in :numref:`PacketNum` should only be used in applications which use wavelet function 'db10'.

.. _PacketNum_MATLAB:

.. figure:: WPT_Tree_with_MATLAB_Numbering.png
   :align: center

   Frequency and natural ordering of wavelet packets in MATLAB