function Freq_Features= WPT_Frequency_Domain_Features(list_name,L,stickout_length,Fs)
%% Frequency Domain Feature Computation
% Description: This code computes the frequency domain features for
%              reconstructed informative wavelet packets and saves them
%              into a .mat file.
% 
% INPUT:
%       list_name      : .txt file that includes the names of the time 
%                        series for each of the stickout size (overhang length)
%       L              : Wavelet Packet Decomposition Level
%       stickout_length: Overhang length of the cutting tool 
%       Fs             : Sampling frequency of the reconstructed time series
%
% OUTPUT: Frequency features array with size of Nx4 where N is the number
%         of reconstructed time series
%
% Note:   Change the file paths and the saving path inside of the function to be able to access
%         to cutting test data and to save data into desired folder
%
% References
% 1) M. C. Yesilli, F. A. Khasawneh, and A. Otto, “On transfer learning for chatter detection in turning
% using wavelet packet transform and empirical mode decomposition,” CIRP J. Manuf. Sci. Technol. 2019, in press.
% 2)F. Khasawneh, A. Otto, and M.C. Yesilli, “Turning dataset for chatter diagnosis using machine learning.”
% Mendeley Data, v1. http://dx.doi.org/10.17632/hvm4wh3jzx.1, 2019.


%add paths for the reconstructed time series
if stickout_length==2
    addpath('D:\Data Archive\Cutting_Test_Data_Documented\cutting_tests_processed\2inch_stickout');
elseif stickout_length==2.5
    addpath('D:\Data Archive\Cutting_Test_Data_Documented\cutting_tests_processed\2p5inch_stickout');
elseif stickout_length==3.5
    addpath('D:\Data Archive\Cutting_Test_Data_Documented\cutting_tests_processed\3p5inch_stickout');
elseif stickout_length==4.5
    addpath('D:\Data Archive\Cutting_Test_Data_Documented\cutting_tests_processed\4p5inch_stickout');
end

%add path for saving the feature matrix 
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
    if stickout_length==2
        sl='2';
    elseif stickout_length==2.5
        sl='2p5';
    elseif stickout_length==3.5
        sl='3p5';
    elseif stickout_length==4.5
        sl='4p5';
    end
    dataname = sprintf('WPT_Level%d_Recon_%sinch_%s',L,sl,filename{(i)});

    time_series{i}=load(dataname);

    eval(sprintf('ts%d=time_series{1,%d}.recon(:,1);',i,i));
    eval(sprintf("l = length(ts%d);",i)); % number of points in the signal

    eval(sprintf("X = ts%d;",i));

    %% take the fft of the signal
    NFFT = 2^nextpow2(l);  % Next power of 2 from length of X
    Y = fft(X, NFFT);
    % number of unique frequency points
    NumUniquePts = ceil((NFFT+1)/2); 
    % grab only the unique components of Y
    Y = Y(1:NumUniquePts);
    MX=abs(Y)/l;  % take magnitude of Y and divide by L to get the linear spectral density

    % multiply by 2 to account for throwing out the second half of the spectrum
    MX = MX * 2;
    % However, the DC component of the signal is unique and should not be 
    %multiplied by 2:
    % Y(1) is the DC component of x for both ODD and EVEN NFFT.
    % Also, for odd NFFT, the Nyquist frequency component is not evaluated, 
    % and the number of unique points is (NFFT+1)/2.
    MX(1) = MX(1)/2; 
    % If NFFT is even, then the FFT will be symmetric such that 
    % the first (1+NFFT/2) points are unique, and FFTX(1+NFFT/2) is 
    % the Nyquist frequency component of x, which is unique. The rest
    % are symmetrically redundant.
    % when NFFT is even, FFTX(1+NFFT/2) is the Nyquist 
    %frequency component of x, and we need to make sure it is unique.
    if ~rem(NFFT, 2)  
         MX(length(MX))=MX(length(MX))/2;
    end 
    f = Fs*(0:(l/2))/l;
    f = Fs/2*linspace(0, 1,  NumUniquePts);
    %% feature computation
    %compute features in frequency domain

    %mean square frequency
    f = transpose(f);
    f_square = f.^2;
    A_MSF(i) = sum(f_square.*MX);
    B_MSF(i) = sum(MX);
    MSF(i) = A_MSF(i)/B_MSF(i);

    %One-step autocorrelation function
    delta_t = 1/10000;
    A_ro(i) = sum(cos(2*pi.*f*delta_t).*MX);
    ro(i) = A_ro(i) / B_MSF(i);

    %Frequency center
    A_FC(i) = sum(f.*MX);
    FC(i) = A_FC(i) / B_MSF(i);
    FC = transpose(FC);
    
    %Standard Frequency 
    f_fc_square = (f - FC(i)).^2;
    A_FV(i) = sum(f_fc_square.*MX);
    FV(i) = A_FV(i)/ B_MSF(i);
    
end

Freq_Features(:,1) = MSF;
Freq_Features(:,2) = ro;
Freq_Features(:,3) = FC;
Freq_Features(:,4) = FV;

savename = sprintf('WPT_Level%d_Freq_Features_%sinch.mat',L,sl);
save_direc = fullfile(save_path,savename);
eval(sprintf("save('%s','Freq_Features');",save_direc));
end
