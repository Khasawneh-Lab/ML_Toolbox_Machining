%load mat file

%2inch case
tic
namets = ["c_320_005","c_425_020","c_425_025","c_570_001","c_570_002","c_570_005","c_570_010","c_770_001","c_770_002_2","c_770_002","c_770_005","c_770_010","i_320_005","i_320_010","i_425_020","i_425_025","i_570_002","i_570_005","i_570_010","i_770_001","s_320_005","s_320_010","s_320_015","s_320_020_2","s_320_020","s_320_025","s_320_030","s_320_035","s_320_040","s_320_045","s_320_050_2","s_320_050","s_425_005","s_425_010","s_425_015","s_425_017","s_425_020","s_570_002","s_570_005"];
for i=1:length(namets)
dataname = sprintf('%s_downsampled.mat',namets(i));
tseries=load(dataname);

%define timeseries
ts = tseries.tsDS(:,2);
t1 = tseries.tsDS(:,1);

%Wavelet Packet Level
L=3

%Wavelet packet decomposition and energy computation
T1 = wpdec(ts,L,'db10');

% Reconstruct 
recon = wprcoef(T1,8);

savename = sprintf('WPT_Level3_Recon_%s.mat',namets(i));
save(savename,'recon');
end
toc
%% FFT of reconstructed signal

% define the signal parameters
Fs = 10000;            % Sampling frequency                    
T = 1/Fs;             % Sampling period 

S=recon;
L = length(S);        % number of points in the signal
t = t1;        % Time vector

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

ticks_fontsize = 23
figure(1)
subplot(2,2,1)
plot(t1,recon)
xmax=max(t1)
xmin=min(t1)
xlim([xmin,xmax])
title('Reconstructed Signal','interpreter', 'latex', 'FontSize', 23)
ylabel('Amplitude $(\frac{m}{s^2})$','interpreter', 'latex', 'FontSize', 23)
xlabel('Time(s)','interpreter', 'latex', 'FontSize', 23)
ax=gca()
set(ax, 'TickLabelInterpreter', 'Latex', ...
    'FontSize', ticks_fontsize);
f = Fs/2*linspace(0, 1,  NumUniquePts);
subplot(2,2,2)
plot(f, MX) 
xmax=max(f)
xmin=min(f)
xlim([xmin,xmax])
ylabel('$|X(f)|$','interpreter', 'latex', 'FontSize', 23)
xlabel('Frequency (Hz)','interpreter', 'latex', 'FontSize', 23)
ax=gca()
set(ax, 'TickLabelInterpreter', 'Latex', ...
    'FontSize', ticks_fontsize);



tseries=load('c_570_002_downsampled.mat');

%define timeseries
ts = tseries.tsDS(:,2);
t1_1 = tseries.tsDS(:,1);
T2 = wpdec(ts,2,'db10');
recon1_1 = wprcoef(T2,3)
recon_1 = recon1_1;

% define the signal parameters
Fs = 10000;            % Sampling frequency                    
T = 1/Fs;             % Sampling period 

S=recon_1;
L = length(S);        % number of points in the signal
t = t1_1;        % Time vector

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
% f = Fs*(0:(L/2))/L;


subplot(2,2,3)
plot(t1_1,recon_1)
xmax=max(t1_1)
xmin=min(t1_1)
xlim([xmin,xmax])
title('Reconstructed Signal','interpreter', 'latex', 'FontSize', 23)
ylabel('Amplitude $(\frac{m}{s^2})$','interpreter', 'latex', 'FontSize', 23)
xlabel('Time(s)','interpreter', 'latex', 'FontSize', 23)
ax=gca()
set(ax, 'TickLabelInterpreter', 'Latex', ...
    'FontSize', ticks_fontsize);
f = Fs/2*linspace(0, 1,  NumUniquePts);
subplot(2,2,4)
plot(f(100:end), MX(100:end)) 
xmax=max(f)
xmin=min(f)
xlim([xmin,xmax])
ylabel('$|X(f)|$','interpreter', 'latex', 'FontSize', 23)
xlabel('Frequency (Hz)','interpreter', 'latex', 'FontSize', 23)
ax=gca()
set(ax, 'TickLabelInterpreter', 'Latex', ...
    'FontSize', ticks_fontsize);

fig1=gcf
save2pdf('C:\Users\yesillim.ME653\Desktop\2inch_320_005_570_002_Chatter_WPT_Level2_Packet_1_ReconFFT',fig1,600)
