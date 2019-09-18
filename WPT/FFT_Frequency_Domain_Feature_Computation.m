%This code take reconstructed signals as input to take their FFT and
%compute the features in frequency domain

%1/10/2019

%Melih Can Yesilli

%2 inch Stickout case

%% load the reconstructed signals
tic
%name list for 2 inch case
namets = ["c_320_005","c_425_020","c_425_025","c_570_001","c_570_002","c_570_005","c_570_010","c_770_001","c_770_002_2","c_770_002","c_770_005","c_770_010","i_320_005","i_320_010","i_425_020","i_425_025","i_570_002","i_570_005","i_570_010","i_770_001","s_320_005","s_320_010","s_320_015","s_320_020_2","s_320_020","s_320_025","s_320_030","s_320_035","s_320_040","s_320_045","s_320_050_2","s_320_050","s_425_005","s_425_010","s_425_015","s_425_017","s_425_020","s_570_002","s_570_005"];

for i=1:length(namets)
ts_name = sprintf('WPT_Level3_Recon_%s',namets(i));
ts_name_time = sprintf('%s_downsampled',namets(i));
ts = load(ts_name);
time = load(ts_name_time);
ts = ts.recon(:,1);
time = time.tsDS(:,1);

%FFT of reconstructed signal

% define the signal parameters
Fs = 10000;            % Sampling frequency                    
T = 1/Fs;             % Sampling period 

S = ts;
L = length(S);   % number of points in the signal
t = time;        % Time vector

% Corrupt the signal with zero-mean white noise with a standard dev. of stdev.
stdev = 0;
X = S;

% % pad the singal with zeros so that its length is a power of 2
% X = [X, zeros(1, 2^(nextpow2(length(X) - length(X))))];

% take the fft of the signal
NFFT = 2^nextpow2(length(X));  % Next power of 2 from length of X
Y = fft(X, NFFT);

% number of unique frequency points
NumUniquePts = ceil((NFFT+1)/2); 

% grab only the unique components of Y
Y = Y(1:NumUniquePts);
MX=abs(Y)/L;  % take magnitude of Y and divide by L to get the linear spectral density

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

% Define the frequency domain f and plot the single-sided amplitude 
% spectrum P1. The amplitudes are not exactly at 0.7 and 1, as expected, 
% because of the added noise. On average, longer signals produce better
% frequency approximations.
f = Fs*(0:(L/2))/L;
f = Fs/2*linspace(0, 1,  NumUniquePts);

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

save('Freq_Features_2inch_WPT_Level3.mat','Freq_Features')
toc