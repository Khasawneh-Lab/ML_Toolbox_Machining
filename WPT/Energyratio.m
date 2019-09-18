% this code take downsampled data and compute energy ratios of wavelet
% packet and then it plots energy ratio graphs

%It also takes FFT of each wavelet packet and plots them 

%320_005 case
%load mat file
str = ['s','i','c'];
for i=1:3
dataname = sprintf('%s_320_005_downsampled.mat',str(i));
ts{i}=load(dataname);
end
%define timeseries
ts1 = ts{1,1}.tsDS(:,2);
t1 = ts{1,1}.tsDS(:,1);
ts2 = ts{1,2}.tsDS(:,2);
t2 = ts{1,2}.tsDS(:,1);
ts3 = ts{1,3}.tsDS(:,2); 
t3 = ts{1,3}.tsDS(:,1);
%Wavelet Level
L=4
%Wavelet packet decomposition and energy computation
T1 = wpdec(ts1,L,'db10')
MaxL1 = wmaxlev(length(ts1),'db10') 
E1 = wenergy(T1);
T2 = wpdec(ts2,L,'db10');
MaxL2 = wmaxlev(length(ts2),'db10') 
E2 = wenergy(T2);
T3 = wpdec(ts3,L,'db10');
MaxL3 = wmaxlev(length(ts3),'db10') 
E3 = wenergy(T3);
%find the energy ratio of wavelet packets
TotalEnergy1= sum(E1);
TotalEnergy2= sum(E2);
TotalEnergy3= sum(E3);
energyratio1 = E1/100; 
energyratio2 = E2/100;
energyratio3 = E3/100;

%Total Wavelet packet number
N=2^L

%plot energy ratios
bpcombined = [transpose(energyratio1), transpose(energyratio2), transpose(energyratio3)];
binedges=(1:N);
figure(1)
subplot(2,1,1)
h = bar(binedges,bpcombined,'grouped')

label_fontsize = 18;
ticks_fontsize = 18;

% add a legend and format it
legend('stable','intermediate chatter','chatter');
set(legend, 'interpreter', 'latex', 'FontSize', 18);

% format the ticks
ax=gca()
set(ax, 'TickLabelInterpreter', 'Latex', ...
    'FontSize', ticks_fontsize);

% set x and y labels and formatting
ax.XLabel.String = 'Wavelet Packets';
ax.XLabel.FontSize = label_fontsize;
ax.XLabel.Interpreter = 'Latex';

ax.YLabel.String = 'Energy Ratios';
ax.YLabel.Interpreter = 'Latex';
ax.YLabel.FontSize = label_fontsize;

%% 570_002 2 inch case
for i=1:3
dataname = sprintf('%s_570_002_downsampled.mat',str(i));
tsecond{i}=load(dataname);
end
%define timeseries
ts_1 = tsecond{1,1}.tsDS(:,2);
t_1 = tsecond{1,1}.tsDS(:,1);
ts_2 = tsecond{1,2}.tsDS(:,2);
t_2 = tsecond{1,2}.tsDS(:,1);
ts_3 = tsecond{1,3}.tsDS(:,2); 
t_3 = tsecond{1,3}.tsDS(:,1);
%Wavelet packet decomposition and energy computation
T1_2 = wpdec(ts_1,L,'db10');
E1_2 = wenergy(T1_2);
T2_2 = wpdec(ts_2,L,'db10');
E2_2 = wenergy(T2_2);
T3_2 = wpdec(ts_3,L,'db10');
E3_2 = wenergy(T3_2);
%find the energy ratio of wavelet packets
TotalEnergy1= sum(E1_2);
TotalEnergy2= sum(E2_2);
TotalEnergy3= sum(E3_2);
energyratio1 = E1_2/100; 
energyratio2 = E2_2/100; 
energyratio3 = E3_2/100; 
%plot energy ratios
bpcombined = [transpose(energyratio1), transpose(energyratio2), transpose(energyratio3)];
binedges=(1:N)
subplot(2,1,2)
h = bar(binedges,bpcombined,'grouped')
label_fontsize = 18;
ticks_fontsize = 18;

% add a legend and format it
legend('stable','intermediate chatter','chatter')
set(legend, 'interpreter', 'latex', 'FontSize', 18);
fig=gcf;

% format the ticks
ax=gca()
set(ax, 'TickLabelInterpreter', 'Latex', ...
    'FontSize', ticks_fontsize);

% set x and y labels and formatting
ax.XLabel.String = 'Wavelet Packets';
ax.XLabel.FontSize = label_fontsize;
ax.XLabel.Interpreter = 'Latex';

ax.YLabel.String = 'Energy Ratios';
ax.YLabel.Interpreter = 'Latex';
ax.YLabel.FontSize = label_fontsize;

fig=gcf;
save2pdf('C:\Users\yesillim.ME653\Desktop\WPT\2inch_320_005-570_002_WPT_Energy_Ratios(Level-3)',fig,600)

%% reconstruct wavelet packets for 570_002
for i=1:2^L
recon_1{i} = wprcoef(T1_2,i+6);
recon_2{i} = wprcoef(T2_2,i+6);
recon_3{i} = wprcoef(T3_2,i+6);

%stable
figure(2)
f1=subplot(2,4,i)
plot(t_1,recon_1{i})
xmax=max(t_1)
xmin=min(t_1)
xlim([xmin,xmax])
k=sprintf('Packet %d',i)
title(k,'interpreter', 'latex', 'FontSize', 12)
ax=gca()
set(ax, 'TickLabelInterpreter', 'Latex', ...
    'FontSize', ticks_fontsize);
% if (i-14)<13
%     set(gca,'XTick',[])
% end
fig1=gcf;

%intermediate chatter
figure(3)
f2=subplot(2,4,i)
plot(t_2,recon_2{i})
xmax=max(t_2)
xmin=min(t_2)
xlim([xmin,xmax])
k=sprintf('Packet %d',i)
title(k,'interpreter', 'latex', 'FontSize', 12)
ax=gca()
set(ax, 'TickLabelInterpreter', 'Latex', ...
    'FontSize', ticks_fontsize);
% if (i-14)<13
%     set(gca,'XTick',[])
% end
fig2=gcf;

%chatter
figure(4)
f3=subplot(2,4,i)
plot(t_3,recon_3{i})
xmax=max(t_3)
xmin=min(t_3)
xlim([xmin,xmax])
sprintf('Packet %d',i)
title(k,'interpreter', 'latex', 'FontSize', 12)
ax=gca()
set(ax, 'TickLabelInterpreter', 'Latex', ...
    'FontSize', ticks_fontsize);
if (i-14)<13
    set(gca,'XTick',[])
end
fig3=gcf;
end
YLabel1H = get(f1,'YLabel')
set(YLabel1H,'String','Amplitude $(\frac{m}{s^2})$');
set(YLabel1H,'Position',[6.68 18e-03 -1]);
set(YLabel1H,'Interpreter', 'Latex', 'fontsize', 18)

XLabel1H = get(f1,'XLabel')
set(XLabel1H,'String','Time(s)');
set(XLabel1H,'Position',[8.7 -0.0068 -1]);
set(XLabel1H,'Interpreter', 'Latex', 'fontsize', 18)

YLabel2H = get(f2,'YLabel')
set(YLabel2H,'String','Amplitude $(\frac{m}{s^2})$');
set(YLabel2H,'Position',[-2.5 12e-03 -1]);
set(YLabel2H,'Interpreter', 'Latex', 'fontsize', 18)

XLabel2H = get(f2,'XLabel')
set(XLabel2H,'String','Time(s)');
set(XLabel2H,'Position',[1.8 -0.0055 -1]);
set(XLabel2H,'Interpreter', 'Latex', 'fontsize', 18)

YLabel3H = get(f3,'YLabel')
set(YLabel3H,'String','Amplitude $(\frac{m}{s^2})$');
set(YLabel3H,'Position',[-12 22e-03 -1.0000]);
set(YLabel3H,'Interpreter', 'Latex', 'fontsize', 18)

XLabel3H = get(f3,'XLabel')
set(XLabel3H,'String','Time(s)');
set(XLabel3H,'Position',[-0.5 -0.0070 -1.0000]);
set(XLabel3H,'Interpreter', 'Latex', 'fontsize', 18)

% save2pdf('C:\Users\yesillim.ME653\Desktop\2inch_570_002_Stable_WPT_Reconstruction',fig1,600)
% save2pdf('C:\Users\yesillim.ME653\Desktop\2inch_570_002_Intermediate_WPT_Reconstruction',fig2,600)
% save2pdf('C:\Users\yesillim.ME653\Desktop\2inch_570_002_Chatter_WPT_Reconstruction',fig3,600)
%% reconstruct wavelet packets for 320_005
for i=1:N
recon1{i} = wprcoef(T1,i+6);
recon2{i} = wprcoef(T2,i+6);
recon3{i} = wprcoef(T3,i+6);

%stable
figure(5)
f1=subplot(2,4,i)
plot(t1,recon1{i})
xmax=max(t1)
xmin=min(t1)
xlim([xmin,xmax])
k=sprintf('Packet %d',i)
title(k,'interpreter', 'latex', 'FontSize', 12)
ax=gca()
set(ax, 'TickLabelInterpreter', 'Latex', ...
    'FontSize', ticks_fontsize);
% if (i-14)<13
%     set(gca,'XTick',[])
% end
fig1=gcf;

%intermediate chatter
figure(6)
f2=subplot(2,4,i)
plot(t2,recon2{i})
xmax=max(t2)
xmin=min(t2)
xlim([xmin,xmax])
k=sprintf('Packet %d',i)
title(k,'interpreter', 'latex', 'FontSize', 12)
ax=gca()
set(ax, 'TickLabelInterpreter', 'Latex', ...
    'FontSize', ticks_fontsize);
% if (i-14)<13
%     set(gca,'XTick',[])
% end
fig2=gcf;

%chatter
figure(7)
f3=subplot(2,4,i)
plot(t3,recon3{i})
xmax=max(t3)
xmin=min(t3)
xlim([xmin,xmax])
sprintf('Packet %d',i)
title(k,'interpreter', 'latex', 'FontSize', 12)
ax=gca()
set(ax, 'TickLabelInterpreter', 'Latex', ...
    'FontSize', ticks_fontsize);
if (i-14)<13
    set(gca,'XTick',[])
end
fig3=gcf;
end
YLabel1H = get(f1,'YLabel')
set(YLabel1H,'String','Amplitude $(\frac{m}{s^2})$');
set(YLabel1H,'Position',[6.68 18e-03 -1]);
set(YLabel1H,'Interpreter', 'Latex', 'fontsize', 18)

XLabel1H = get(f1,'XLabel')
set(XLabel1H,'String','Time(s)');
set(XLabel1H,'Position',[8.7 -0.0068 -1]);
set(XLabel1H,'Interpreter', 'Latex', 'fontsize', 18)

YLabel2H = get(f2,'YLabel')
set(YLabel2H,'String','Amplitude $(\frac{m}{s^2})$');
set(YLabel2H,'Position',[-2.5 12e-03 -1]);
set(YLabel2H,'Interpreter', 'Latex', 'fontsize', 18)

XLabel2H = get(f2,'XLabel')
set(XLabel2H,'String','Time(s)');
set(XLabel2H,'Position',[1.8 -0.0055 -1]);
set(XLabel2H,'Interpreter', 'Latex', 'fontsize', 18)

YLabel3H = get(f3,'YLabel')
set(YLabel3H,'String','Amplitude $(\frac{m}{s^2})$');
set(YLabel3H,'Position',[-12 22e-03 -1.0000]);
set(YLabel3H,'Interpreter', 'Latex', 'fontsize', 18)

XLabel3H = get(f3,'XLabel')
set(XLabel3H,'String','Time(s)');
set(XLabel3H,'Position',[-0.5 -0.0070 -1.0000]);
set(XLabel3H,'Interpreter', 'Latex', 'fontsize', 18)

% save2pdf('C:\Users\yesillim.ME653\Desktop\2inch_570_002_Stable_WPT_Reconstruction',fig1,600)
% save2pdf('C:\Users\yesillim.ME653\Desktop\2inch_570_002_Intermediate_WPT_Reconstruction',fig2,600)
% save2pdf('C:\Users\yesillim.ME653\Desktop\2inch_570_002_Chatter_WPT_Reconstruction',fig3,600)




%% FFT of reconstructed signals for 320_005
for i=1:N
% define the signal parameters
Fs = 10000;            % Sampling frequency                    
T = 1/Fs;             % Sampling period 

S=recon_3{i};
L = length(S); % number of points in the signal
t = t_3;        % Time vector

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
figure(8)
h2 = subplot(2, 4, i);
f = Fs/2*linspace(0, 1,  NumUniquePts);
plot(f, MX) 
xmax=max(f)
xmin=min(f)
xlim([xmin,xmax])

% if i<13
%     set(gca,'XTick',[])
% end

k=sprintf('Packet %d',i)
title(k,'interpreter', 'latex', 'FontSize', 12)
ax=gca()
set(ax, 'TickLabelInterpreter', 'Latex', ...
    'FontSize', ticks_fontsize);
fig5=gcf
end
YLabel2H = get(h2,'YLabel')
set(YLabel2H,'String','$|X(f)|$');
set(YLabel2H,'Position',[-21000 539e-05 -1]);
set(YLabel2H,'Interpreter', 'Latex', 'fontsize', 18)

XLabel3H = get(h2,'XLabel')
set(XLabel3H,'String','Frequency (Hz)');
set(XLabel3H,'Position',[-7.5e+03 -50e-05 -1.0000]);
set(XLabel3H,'Interpreter', 'Latex', 'fontsize', 18)
save2pdf('C:\Users\yesillim.ME653\Desktop\WPT\2inch_570_002_Chatter_WPT_Level3_Reconstruction_FFT',fig5,300)

text(10, 4*10^(-3),'\bf (d)', 'Interpreter', 'Latex', 'fontsize', 23)