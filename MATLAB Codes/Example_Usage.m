%% Example Usage of Matlab Codes

% References
% 1) M. C. Yesilli, F. A. Khasawneh, and A. Otto, “On transfer learning for chatter detection in turning
% using wavelet packet transform and empirical mode decomposition,” CIRP J. Manuf. Sci. Technol. 2019, in press.
% 2)F. Khasawneh, A. Otto, and M.C. Yesilli, “Turning dataset for chatter diagnosis using machine learning.”
% Mendeley Data, v1. http://dx.doi.org/10.17632/hvm4wh3jzx.1, 2019.

%% Determine the informative Wavelet Packets
clear all; clc;close all;

N=3;
ts_no = [5,17,38];
L=4;
list_name = 'time_series_name_2inch.txt';
WF = 'db10';
stickout_length =2;
help WP_Energy_Ratio
column_no = 4;
row_no = 4;
Fs =10000;

WP_Energy_Ratio(L, stickout_length,list_name,ts_no,WF,N,4,4,Fs)

%% Reconstruct Wavelet Packets based on selected informative packet number
clear all; clc;close all;

list_name = 'time_series_name_2inch.txt';
WF = 'db10';
stickout_length =2;
IWP=3;
L=4;
WPT_Informative_Packet_Recon(list_name,L,WF,IWP,stickout_length)


%% Compute frequency domain features based on selected level of transform 

clear all; clc;close all;
Fs=10000;
list_name = 'time_series_name_4p5inch.txt';
L=4;
stickout_length =4.5;
WPT_Frequency_Domain_Features(list_name,L,stickout_length,Fs)
