function WPT_Informative_Packet_Recon(list_name,L,WF,IWP,stickout_length)
%% Wavelet Packet Reconstruction
% Description: This code computes WPT for given time series and reconstructs
%              the chosen informative wavelet packets and saves them into
%              desired folder. 
% 
% INPUT:
%       list_name      : .txt file that includes the names of the time 
%                        series for each of the stickout size (overhang length)
%       L              : Wavelet Packet Decomposition Level
%       WF             : Wavelet Function
%       IWP            : Informative Wavelet Packet Number
%       stickout_length: Overhang length of the cutting tool 
%
% OUTPUT: Saving reconstructed wavelet packets in time domain into desired
%         location
%
% Note:   Change the file paths and the saving path inside of the function to be able to access
%         to cutting test data and to save data into desired folder
%
% References
% 1) M. C. Yesilli, F. A. Khasawneh, and A. Otto, “On transfer learning for chatter detection in turning
% using wavelet packet transform and empirical mode decomposition,” CIRP J. Manuf. Sci. Technol. 2019, in press.
% 2)F. Khasawneh, A. Otto, and M.C. Yesilli, “Turning dataset for chatter diagnosis using machine learning.”
% Mendeley Data, v1. http://dx.doi.org/10.17632/hvm4wh3jzx.1, 2019.

%add paths for the current data folder for cutting tests
if stickout_length==2
    addpath('D:\Data Archive\Cutting_Test_Data_Documented\cutting_tests_processed\2inch_stickout');
elseif stickout_length==2.5
    addpath('D:\Data Archive\Cutting_Test_Data_Documented\cutting_tests_processed\2p5inch_stickout');
elseif stickout_length==3.5
    addpath('D:\Data Archive\Cutting_Test_Data_Documented\cutting_tests_processed\3p5inch_stickout');
elseif stickout_length==4.5
    addpath('D:\Data Archive\Cutting_Test_Data_Documented\cutting_tests_processed\4p5inch_stickout');
end

%add path for saving the reconstructed time series 
if stickout_length==2
    save_path='D:\Data Archive\Cutting_Test_Data_Documented\cutting_tests_processed\2inch_stickout';
elseif stickout_length==2.5
    save_path='D:\Data Archive\Cutting_Test_Data_Documented\cutting_tests_processed\2p5inch_stickout';
elseif stickout_length==3.5
    save_path='D:\Data Archive\Cutting_Test_Data_Documented\cutting_tests_processed\3p5inch_stickout';
elseif stickout_length==4.5
    save_path='D:\Data Archive\Cutting_Test_Data_Documented\cutting_tests_processed\4p5inch_stickout';
end


filename = regexp(fileread(list_name), '\r?\n', 'split');

for i=1:length(filename)
    %load mat files
    dataname = sprintf('%s',filename{(i)});

    time_series=load(dataname);
   
    eval(sprintf('ts%d=time_series.tsDS(:,2);',i,i));
    eval(sprintf('time_%d=time_series.tsDS(:,1);',i,i));
    
    % Wavelet Packet Decomposition
    eval(sprintf('WPT%d=wpdec(ts%d,%d,"%s");',i,i,L,WF));

    % Reconstruction Wavelet Packets
    eval(sprintf('recon = wprcoef(WPT%d,%d);',i,IWP))
    
    %saving
    if stickout_length==2
        sl='2';
    elseif stickout_length==2.5
        sl='2p5';
    elseif stickout_length==3.5
        sl='3p5';
    elseif stickout_length==4.5
        sl='4p5';
    end
    savename = sprintf('WPT_Level%d_Recon_%sinch_%s',L,sl,dataname);
    save_direc = fullfile(save_path,savename);
    eval(sprintf("save('%s','recon');",save_direc));
end
end




