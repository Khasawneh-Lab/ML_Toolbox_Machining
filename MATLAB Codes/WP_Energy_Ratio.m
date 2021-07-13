function fig = WP_Energy_Ratio(L, stickout_length,list_name,ts_no,WF,N,row_no,column_no,Fs) 
%% Wavelet Packet Decomposition and Energy Ratio Computation
% Description: This code takes time series of cutting tests and computes 
%              their Wavelet Packet Decomposition. Energy ratios of each 
%              wavelet packet, reconstructed wavelet packets in time domain
%              for each transform and FFT of these packets are plotted as outputs.
%              These plots help user to indentify the informative wavelet
%              packet for given time series. 
% 
% INPUT:
%       L              : Wavelet Packet Decomposition Level
%       stickout_length: Overhang length of the cutting tool 
%       list_name      : .txt file that includes the names of the time 
%                        series for each of the stickout size (overhang length)
%       ts_no          : 1 x n array that includes the number of the time series 
%                      : which user wants to investigate their WPT inside of the list 
%       WF             : Wavelet Function
%       N              : number of time series whose WPT will be computed
%       row_no         : row number for the plots that include reconstructed
%                        wavelet packets in time domanin and frequency domain
%       column_no      : column number for the plots that include reconstructed
%                        wavelet packets in time domanin and frequency domain
%       Fs             : sampling frequency of the time series
%
% OUTPUT: - 1st Plot: It shows energy ratio of each packet of the decomposition 
%         for selected time series inside of the list
%         - 2nd to (N+1)th Plot: Plots of reconstructed packets of WPT for each
%         investigated time series in time domain
%         - (N+2)th to (2N+1)th Plot: FFT of reconstructed packets of WPT for each
%         investigated time series 
%
% Note:   Change the file paths inside of the function to be able to access
%         to cutting test data
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


filename = regexp(fileread(list_name), '\r?\n', 'split');

for i=1:N
    %load mat files
    dataname = sprintf('%s',filename{ts_no(i)});

    % extract the time series information from its name
    rpm(i) = str2num(dataname(3:5));
    doc(i) = str2num(sprintf('0.%s',dataname(7:9)));
    label(i) = string(dataname(1));
    time_series{i}=load(dataname);

    eval(sprintf('ts%d=time_series{1,%d}.tsDS(:,2);',i,i));
    eval(sprintf('time_%d=time_series{1,%d}.tsDS(:,1);',i,i));
    
    % Wavelet Packet Decomposition
    eval(sprintf('WPT%d=wpdec(ts%d,%d,"%s");',i,i,L,WF));
    
    %Energy computation for each packet
    eval(sprintf('E%d=wenergy(WPT%d);',i,i));

    %Energy ratios 
    eval(sprintf('energyratio%d=(E%d)/100;',i,i));
end

%Total Wavelet packet number
WP_No=2^L;

%% Plotting

for i=1:N
    eval(sprintf('bpcombined(:,%d) = transpose(energyratio%d);',i,i));
    dataname = sprintf('%s',filename{ts_no(i)});
    if label(i) == 'c'
        legend_names(i)= string(sprintf('Chatter,Stickout=%d inch, RPM=%d, DOC=%.4f',stickout_length,rpm(i),doc(i)));
    elseif label(i) == 's'
        legend_names(i)= string(sprintf('Stable,Stickout=%d inch, RPM=%d, DOC=%.4f',stickout_length,rpm(i),doc(i)));
    elseif label(i) == 'i'
        legend_names(i)= string(sprintf('Mild Chatter,Stickout=%d inch, RPM=%d, DOC=%.4f',stickout_length,rpm(i),doc(i)));
    end
end

%bar plot of energy ratios
binedges=(1:WP_No);
figure(1);
h = bar(binedges,bpcombined,'grouped');

label_fontsize = 18;
ticks_fontsize = 18;

% add a legend and format it
legend(legend_names);
set(legend, 'interpreter', 'latex', 'FontSize', 15);

% format the ticks
ax=gca();
set(ax, 'TickLabelInterpreter', 'Latex', ...
    'FontSize', ticks_fontsize);

% set x and y labels and formatting
ax.XLabel.String = 'Wavelet Packets';
ax.XLabel.FontSize = label_fontsize;
ax.XLabel.Interpreter = 'Latex';

ax.YLabel.String = 'Energy Ratios';
ax.YLabel.Interpreter = 'Latex';
ax.YLabel.FontSize = label_fontsize;
fig = gcf();
%% Reconstruction of each wavelet packet and their plot

%wavelet packet nodes to be used in the 
terminal_nodes_start = 2^(L)-1;
terminal_nodes_end = 2^(L+1)-2;


for i=1:N
  % array name which will include the reconstructed packets inside for each
  % time series
  a_name = sprintf('recon_%d',i);
  eval(sprintf(' time_vec =time_%d;',i))
  %wavelet packet number defined by matlab wavelet tree
  terminal_node = terminal_nodes_start;
  
  figure(i+1)
  
  for j=1:WP_No
    eval(sprintf('%s{%d} = wprcoef(WPT%d,%d);',a_name,j,i,terminal_node));  
    
    % plotting of each reconstructed packets
    h=subplot(row_no,column_no,j);
    pos1(j,1:4)=get(h,'position');
    eval(sprintf('plot(time_%d,%s{%d})',i,a_name,j))
    xmax=max(time_vec);
    xmin=min(time_vec);
    xlim([xmin,xmax]);
    k=sprintf('Packet %d',j);
    title(k,'interpreter', 'latex', 'FontSize', 12);
    ax=gca();
    set(ax, 'TickLabelInterpreter', 'Latex', ...
        'FontSize', ticks_fontsize);
    eval(sprintf('fig_%d=gcf;',i+1)); 
    terminal_node = terminal_node+1;
  end 
  height = (pos1(1,2)+pos1(1,4)-pos1(end,2));
  h3=axes('position',[pos1(1,1)-0.03 pos1(end,2) pos1(end,3) height],'visible','off');
  y_label=ylabel('Amplitude $(\frac{m}{s^2})$','visible','on','interpreter', 'latex', 'FontSize', 20);
  h4=axes('position',[pos1(1,1)+0.31 pos1(end,2)-0.01 pos1(end,3) height],'visible','off');
  x_label=xlabel('Time(s)','visible','on','interpreter', 'latex', 'FontSize', 20);
end
%% FFT for Reconstructed Signals
for i=1:N
    eval(sprintf(' time_vec =time_%d;',i))
    figure(i+N+1)
    for j=1:WP_No
        % define the signal parameters
        T = 1/Fs;             % Sampling period 

        eval(sprintf('time_series = recon_%d{%d};',i,j))
        l = length(time_series); % number of points in the signal

        X=time_series;

        % take the fft of the signal
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
        f = Fs/2*linspace(0, 1,  NumUniquePts);
        %plotting
        h2=subplot(row_no,column_no,j);
        pos2(j,1:4)=get(h,'position');
        plot(f, MX)
        %formatting the figure
        xmax=max(f);
        xmin=min(f);
        xlim([xmin,xmax])
        k=sprintf('Packet %d',j);
        title(k,'interpreter', 'latex', 'FontSize', 12)
        ax=gca();
        set(ax, 'TickLabelInterpreter', 'Latex', ...
            'FontSize', ticks_fontsize);
        eval(sprintf('fig_%d=gcf;',i+1+N));
    end
    height = (pos1(1,2)+pos1(1,4)-pos1(end,2));
    h5=axes('position',[pos2(1,1)-0.65 pos2(end,2) pos2(end,3) height],'visible','off');
    y_label=ylabel('$|X(f)|$','visible','on','interpreter', 'latex', 'FontSize', 20);
    h6=axes('position',[pos2(1,1)-0.32 pos2(end,2)-0.01 pos2(end,3) height],'visible','off');
    x_label=xlabel('Frequency(Hz)','visible','on','interpreter', 'latex', 'FontSize', 20);    
end
end

