function fig=PSD_channel(EEG)
% Compute PSD for each channel
n_chan=EEG.nbchan;
data = EEG.data;
% a 50% overlap
winsize=EEG.srate;
overlap=round(winsize/2);
%creating a matrix to help us check for peaks in data
pxx_matrix = [];
oof_matrix = [];
resids_pxx_matrix = [];

%run through all the components
for ic=1:EEG.nbchan
    [pxx,frex,~,~,~] = spectopo(EEG.data(ic,:),EEG.pnts,...
        EEG.srate, 'winsize', winsize, 'overlap', overlap,...
        'plot', 'off', 'freqfac',2);
    % Get frequencies from 1 to 100
    freq_idx=find((frex>=1) & (frex<100));
    frex=frex(freq_idx);
    pxx=pxx(:,freq_idx);
    
    % Remove 58.8-61.5Hz and either set to NaN or interpolate
    freq_idx=find((frex>=58.8) & (frex<=61.5));
    pxx(freq_idx)=NaN;
    nanx = isnan(pxx);
    t    = 1:numel(pxx);
    pxx(nanx) = interp1(t(~nanx), pxx(~nanx), t(nanx));
    
    pxx_matrix(ic,:)=pxx;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Get the residual values
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    oof=log10(1./frex);
    % Fit 1/f to spectrum
    lm_psd=fitlm(oof,pxx,'RobustOpts','on');
    % Get fitted 1/f function
    fitted=lm_psd.Fitted;
    oof_matrix(ic,:)=fitted;
    resids_pxx=lm_psd.Residuals.Raw;
    resids_pxx_matrix(ic,:)=resids_pxx;
end

fig=figure();
subplot(2,1,1);
hold all;
 for ic=1:EEG.nbchan
    plot(frex,pxx_matrix(ic,:))
end
ylabel('Power');
xlabel('Frequency (Hz)');
subplot(2,1,2);
hold all
for ic=1:EEG.nbchan
    plot(frex,resids_pxx_matrix(ic,:));
end
ylabel('Power');
xlabel('Frequency (Hz)');


