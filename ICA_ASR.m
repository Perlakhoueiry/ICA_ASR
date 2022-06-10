function ICA_ASR(study_info)

%% Clear variable space and run eeglab

addpath('NoiseTools');
addpath(genpath('C:\Users\khoueiry\Desktop\eeglab2021.1'));

ext='.set';

% Enter the path of the channel location file
channel_locations = 'C:\Users\khoueiry\Desktop\dev_beta_umd\data\GSN-HydroCel-65.sfp';

% Do your data need correction for anti-aliasing filter and/or task related time offset?
% 0 = NO (no correction), 1 = YES (correct time offset)
adjust_time_offset = 1;
% If your data need correction for time offset, initialize the offset time (in milliseconds)
% anti-aliasing time offset (in milliseconds). 0 = No time offset
filter_timeoffset = 18;
% stimulus related time offset (in milliseconds). 0 = No time offset
stimulus_timeoffset = 0;
% response related time offset (in milliseconds). 0 = No time offset
response_timeoffset = 0;
% enter the stimulus makers that need to be adjusted for time offset
stimulus_markers = {};
% enter the response makers that need to be adjusted for time offset
response_markers = {};

% Do you want to down sample the data?
% 0 = NO (no down sampling), 1 = YES (down sampling)
down_sample = 0;
% set sampling rate (in Hz), if you want to down sample
sampling_rate = 0;

% Do you want to delete the outer layer of the channels? (Rationale
% has been described in MADE manuscript)
%    This function can also be used to down sample electrodes. For example,
%    if EEG was recorded with 128 channels but you would like to analyse
%    only 64 channels, you can assign the list of channnels to be excluded
%    in the 'outerlayer_channel' variable.
% 0 = NO (do not delete outer layer), 1 = YES (delete outerlayer);
delete_outerlayer = 0;
% recommended list for EGI 128 channel net: {'E17' 'E38' 'E43' 'E44' 'E48'
% 'E49' 'E113' 'E114' 'E119' 'E120' 'E121' 'E125' 'E126' 'E127' 'E128'
% 'E56' 'E63' 'E68' 'E73' 'E81' 'E88' 'E94' 'E99' 'E107'}

% Initialize the filters
% High-pass frequency
highpass = 1;
% Low-pass frequency. We recommend low-pass filter at/below line noise
% frequency (see manuscript for detail)
lowpass  = 100;

% Are you processing task-related or resting-state EEG data?
% 0 = resting, 1 = task
task_eeg = 1;
% enter all the event/condition markers
task_event_markers = {'OBM','EBM','FTGO','FTGE'};

% 9. Do you want to epoch/segment your data?
% 0 = NO (do not epoch), 1 = YES (epoch data)
epoch_data = 1;
% epoch length in second
task_epoch_length = [-1.5 1.5];
% for resting EEG continuous data will be segmented into consecutive epochs
% of a specified length (here 2 second) by adding dummy events
%rest_epoch_length = xx;
% 0 = NO (do not create overlapping epoch), 1 = YES (50% overlapping epoch)
%overlap_epoch = 0;
%dummy_events ={'xxx'}; % enter dummy events name

% Do you want to remove/correct baseline?
% 0 = NO (no baseline correction), 1 = YES (baseline correction)
remove_baseline = 0;
% baseline period in milliseconds (MS) [] = entire epoch
baseline_window = [];

% Do you want to remove artifact laden epoch based on voltage threshold?
% 0 = NO, 1 = YES
voltthres_rejection = 1;
% lower and upper threshold (in mV)
volt_threshold = [-150 150];

% Do you want to perform epoch level channel interpolation for artifact
% laden epoch? (see manuscript for detail)
% 0 = NO, 1 = YES.
interp_epoch = 1; 
% If you set interp_epoch = 1, enter the list of frontal channels to check
% (see manuscript for detail)
frontal_channels = {'E1', 'E5', 'E10', 'E17'}; 
% recommended list for EGI 128 channel net: {'E1', 'E8', 'E14', 'E21',
% 'E25', 'E32', 'E17'}

% Do you want to save interim results?
% 0 = NO (Do not save) 1 = YES (save interim results)
save_interim_result = 1;

% How do you want to save your data? .set or .mat
% 1 = .set (EEGLAB data structure), 2 = .mat (Matlab data structure)
output_format = 1;

%% Define User-Parameters here

% Please read the parameters.doc file for more information: LINK to be
% provided
params.isSegt   = 0; % set to 0 if you do not want to segment the data based on newborn's visual attention for the presented stimuli
params.isBadCh  = 1; % set to 1 if you want to employ NEAR Bad Channel Detection
params.isBadSeg = 1; % set to 1 if you want to employ NEAR Bad Epochs Rejection/Correction (using ASR)
params.isVisIns = 0; % set to 1 if you want to visualize intermediate cleaning of NEAR Cleaning (bad channels + bad segments)
params.isInterp = 1; % set to 1 if you want to interpolate the removed bad channels (by Spherical Interpolation)
params.isAvg    = 1; % set to 1 if you want to perform average referencing

% Segmentation using fixation intervals - parameters begin %
% N.B: The following parameters can be set to [] if params.isSegt = 0
params.sname = 'segt_visual_attention.xlsx'; % the visual segmentation coding file
params.sloc  = []; % location of the xlsx file
params.look_thr = 4999; % consider only the segments that exceed this threshold+1 in ms to retain; alternatively can be set to [] if no thresholding is preferred
% Segmentation using fixation intervals - parameters end %

% Parameters for NEAR - Bad Channels Detection begin %
% d) flat channels
params.isFlat  = 1; % flag variable to enable or disable Flat-lines detection method (default: 1)
params.flatWin = 5; % tolerance level in s(default: 5)

% b) LOF (density-based)
params.isLOF       = 1;  % flag variable to enable or disable LOF method (default: 1)
params.dist_metric = 'seuclidean'; % Distance metric to compute k-distance
params.thresh_lof  = 2.5; % Threshold cut-off for outlier detection on LOF scores
params.isAdapt = 10; % The threshold will be incremented by a factor of 1 if the given threshold detects more than xx %
%of total channels (eg., 10); if this variable left empty [], no adaptive thresholding is enabled.


% c) Periodogram (frequency based)
params.isPeriodogram = 0; % flag variable to enable or disable periodogram method (default: 0)
params.frange        = [1 20]; % Frequency Range in Hz
params.winsize       = 1; % window length in s
params.winov         = 0.66; % 66% overlap factor
params.pthresh       = 4.5; % Threshold Factor to predict outliers on the computed energy

% Parameters for NEAR - Bad Channels Detection end %

% Parameters for NEAR- Bad Segments Correction/Rejection using ASR begin %

params.rej_cutoff = 13;   % A lower value implies severe removal (Recommended value range: 20 to 30)
params.rej_mode   = 'off'; % Set to 'off' for ASR Correction and 'on for ASR Removal (default: 'on')
params.add_reject = 'off'; % Set to 'on' for additional rejection of bad segments if any after ASR processing (default: 'off')

% Parameters for NEAR- Bad Segments Correction/Rejection using ASR end %

% Parameter for interpolation begin %

params.interp_type = 'spherical'; % other values can be 'v4'. Please refer to pop_interp.m for more details.

% Parameter for interpolation end %

% Parameter for Re-referencing begin %
params.reref = 30; % if isAvg was set to 0, this parameter must be set.
%params.reref = {'E124'}; % reref can also be the channel name.

% Parameter for Re-referencing begin %

%% Initialize output variables
lof_flat_channels={};
lof_channels={};
lof_periodo_channels={};
% Bad channels identified using LOF
lof_bad_channels={};
% number of bad channel/s due to channel/s exceeding xx% of artifacted epochs
ica_preparation_bad_channels=[];
% length of data (in second) fed into ICA decomposition
length_ica_data=[];
% total independent components (ICs)
total_ICs=[];
% number of artifacted ICs
ICs_removed=[];
% number of epochs before artifact rejection
total_epochs_before_artifact_rejection=[];
% number of epochs after artifact rejection
total_epochs_after_artifact_rejection=[];
% total_channels_interpolated=faster_bad_channels+ica_preparation_bad_channels
total_channels_interpolated=[];
asr_tot_samples_modified=[];
asr_change_in_RMS=[];

%% Loop over all data files
study_info.participant_info=study_info.participant_info(1:2,:);
for s_idx=1:size(study_info.participant_info,1)
    % Get subject ID from study info
    subject=study_info.participant_info.participant_id{s_idx};
    
    % Where original raw data is located
    subject_raw_data_dir=fullfile(study_info.data_dir, 'data',subject, 'eeg');
    
    % Where to put processed (derived) data
    subject_output_data_dir=fullfile(study_info.data_dir, 'derivatives', 'NEARICA', subject);
    
    if save_interim_result ==1
        
        if exist([subject_output_data_dir filesep '01_filtered_data'], 'dir') == 0
            mkdir([subject_output_data_dir filesep '01_filtered_data'])
        end
        
        if exist([subject_output_data_dir filesep '02_near_data'], 'dir') == 0
            mkdir([subject_output_data_dir filesep '02_near_data'])
        end
        
        if exist([subject_output_data_dir filesep '03_ica_data'], 'dir') == 0
            mkdir([subject_output_data_dir filesep '03_ica_data'])
        end
        
        if exist([subject_output_data_dir filesep '04_rereferenced_data'], 'dir') == 0
            mkdir([subject_output_data_dir filesep '04_rereferenced_data'])
        end
    end
    if exist([subject_output_data_dir filesep 'processed_data'], 'dir') == 0
        mkdir([subject_output_data_dir filesep 'processed_data'])
    end
    
    fprintf('\n\n\n*** Processing subject %s ***\n\n\n', subject);
    
    
    % Parameter for Re-referencing begin %
    
    isSegt        = params.isSegt;
    isBadCh       = params.isBadCh;
    isVisIns      = params.isVisIns;
    isBadSeg      = params.isBadSeg;
    isInterp      = params.isInterp;
    isAvg         = params.isAvg;
    
    look_thr      = params.look_thr;
    
    isFlat        = params.isFlat;
    flatWin       = params.flatWin;
    isLOF         = params.isLOF;
    dist_metric   = params.dist_metric;
    thresh_lof    = params.thresh_lof;
    isAdapt       = params.isAdapt;
    isPeriodogram = params.isPeriodogram;
    frange        = params.frange;
    winsize       = params.winsize;
    winov         = params.winov;
    pthresh       = params.pthresh;
    
    rej_cutoff    = params.rej_cutoff;
    rej_mode      = params.rej_mode;
    add_reject    = params.add_reject;
    
    interp_type   = params.interp_type;
    
    reref         = params.reref;
    
    %% Step 2a: Import data
    data_file_name=sprintf('%s_task-%s_eeg.set',subject, study_info.task);
    EEG=pop_loadset('filename', data_file_name, 'filepath', subject_raw_data_dir);
    EEG = eeg_checkset(EEG);
    origEEG=EEG;
    
    %% STEP 1.5: Delete discontinuous data from the raw data file
    % (OPTIONAL, but necessary for most EGI files)
    % Note: code below may need to be modified to select the appropriate
    % markers (depends on the import function) remove discontinous data at
    % the start and end of the file
    % boundary markers often indicate discontinuity
    disconMarkers = find(strcmp({EEG.event.type}, 'boundary'));
    if length(disconMarkers)>0
        % remove discontinuous chunk
        EEG = eeg_eegrej( EEG, [1 EEG.event(disconMarkers(1)).latency] );
        EEG = eeg_checkset( EEG );
    end
    % remove data after last task event (OPTIONAL for EGI files... useful when
    % file has noisy data at the end)
    end_base_flags=find(strcmp({EEG.event.type},'LBOB'));
    end_exp_flags=find(strcmp({EEG.event.type},'LBEX'));
    latency=max([EEG.event(end_base_flags(end)).latency EEG.event(end_exp_flags(end)).latency]);
    % remove everything 1.5 seconds after the last event
    EEG = eeg_eegrej( EEG, [(latency+(1.5*EEG.srate)) EEG.pnts] );
    EEG = eeg_checkset( EEG );
    
    lbse_flags = find(strcmp({EEG.event.type},'LBSE'));
    lext_flags = find(strcmp({EEG.event.type},'LEXT'));
    latency=max([EEG.event(lbse_flags(1)).latency EEG.event(lext_flags(1)).latency]);
    % remove everything until 1.5 seconds before the first event
    EEG = eeg_eegrej( EEG, [1 (latency-(1.5*EEG.srate))] );
    EEG = eeg_checkset( EEG );
    
    %% STEP 2: Import channel locations
    EEG=pop_chanedit(EEG, 'load',{channel_locations 'filetype' 'autodetect'});
    EEG = eeg_checkset( EEG );
    
    % Check whether the channel locations were properly imported. The EEG
    % signals and channel numbers should be same.
    if size(EEG.data, 1) ~= length(EEG.chanlocs)
        error('The size of the data does not match with channel numbers.');
    end
    
    % Plot channel layout
    fig=figure();
    topoplot([],EEG.chanlocs, 'style', 'blank',  'electrodes', 'labelpoint', 'chaninfo', EEG.chaninfo);
    saveas(fig, fullfile(subject_output_data_dir,'01-initial_ch_locations.png'));
    
    %% STEP 2.5: Label the task (OPTIONAL)
    % Add observation baseline events
    base_obs_latencies=[EEG.event(find(strcmp('LBOB',{EEG.event.type}))).latency];
    for i=1:length(base_obs_latencies)
        n_events=length(EEG.event);
        EEG.event(n_events+1).type='OBM';
        EEG.event(n_events+1).latency=(base_obs_latencies(i)-1.5*EEG.srate)-1;
        EEG.event(n_events+1).urevent=n_events+1;
    end
    %check for consistency and reorder the events chronologically...
    EEG=eeg_checkset(EEG,'eventconsistency');
    
    % Add execution baseline events
    exe_obs_latencies=[EEG.event(find(strcmp('LBEX',{EEG.event.type}))).latency];
    for i=1:length(exe_obs_latencies)
        n_events=length(EEG.event);
        EEG.event(n_events+1).type='EBM';
        EEG.event(n_events+1).latency=(exe_obs_latencies(i)-1.5*EEG.srate)-1;
        EEG.event(n_events+1).urevent=n_events+1;
    end
    %check for consistency and reorder the events chronologically...
    EEG=eeg_checkset(EEG,'eventconsistency');
    
    %% STEP 3: Adjust anti-aliasing and task related time offset
    if adjust_time_offset==1
        % adjust anti-aliasing filter time offset
        if filter_timeoffset~=0
            for aafto=1:length(EEG.event)
                EEG.event(aafto).latency=EEG.event(aafto).latency+(filter_timeoffset/1000)*EEG.srate;
            end
        end
        % adjust stimulus time offset
        if stimulus_timeoffset~=0
            for sto=1:length(EEG.event)
                for sm=1:length(stimulus_markers)
                    if strcmp(EEG.event(sto).type, stimulus_markers{sm})
                        EEG.event(sto).latency=EEG.event(sto).latency+(stimulus_timeoffset/1000)*EEG.srate;
                    end
                end
            end
        end
        % adjust response time offset
        if response_timeoffset~=0
            for rto=1:length(EEG.event)
                for rm=1:length(response_markers)
                    if strcmp(EEG.event(rto).type, response_markers{rm})
                        EEG.event(rto).latency=EEG.event(rto).latency-(response_timeoffset/1000)*EEG.srate;
                    end
                end
            end
        end
    end
    
    %% STEP 4: Change sampling rate
    if down_sample==1
        if floor(sampling_rate) > EEG.srate
            error ('Sampling rate cannot be higher than recorded sampling rate');
        elseif floor(sampling_rate) ~= EEG.srate
            EEG = pop_resample( EEG, sampling_rate);
            EEG = eeg_checkset( EEG );
        end
    end
    
    %% STEP 5: Delete outer layer of channels
    chans_labels=cell(1,EEG.nbchan);
    for i=1:EEG.nbchan
        chans_labels{i}= EEG.chanlocs(i).labels;
    end
    [chans,chansidx] = ismember(study_info.outerlayer_channel, chans_labels);
    outerlayer_channel_idx = chansidx(chansidx ~= 0);
    if delete_outerlayer==1
        if isempty(outerlayer_channel_idx)==1
            error(['None of the outer layer channels present in channel locations of data.'...
                ' Make sure outer layer channels are present in channel labels of data (EEG.chanlocs.labels).']);
        else
            fig=compute_and_plot_psd(EEG,outerlayer_channel_idx);
            saveas(fig, fullfile(subject_output_data_dir,'02-outer_ch_psd.png'));
            
            EEG = pop_select( EEG,'nochannel', outerlayer_channel_idx);
            EEG = eeg_checkset( EEG );
        end
    end
    
    % Plot channel locations
    fig=figure();
    topoplot([],EEG.chanlocs, 'style', 'blank',  'electrodes', 'labelpoint', 'chaninfo', EEG.chaninfo);
    saveas(fig, fullfile(subject_output_data_dir,'03-inner_ch_locations.png'));
    
    % Plot PSD
    fig=compute_and_plot_psd(EEG, 1:EEG.nbchan);
    saveas(fig, fullfile(subject_output_data_dir,'04-inner_ch_psd.png'));
    
    
    %% STEP 6: Filter data
    % Calculate filter order using the formula: m = dF / (df / fs), where m = filter order,
    % df = transition band width, dF = normalized transition width, fs = sampling rate
    % dF is specific for the window type. Hamming window dF = 3.3
    
    high_transband = highpass; % high pass transition band
    low_transband = 10; % low pass transition band
    
    hp_fl_order = 3.3 / (high_transband / EEG.srate);
    lp_fl_order = 3.3 / (low_transband / EEG.srate);
    
    % Round filter order to next higher even integer. Filter order is always even integer.
    if mod(floor(hp_fl_order),2) == 0
        hp_fl_order=floor(hp_fl_order);
    elseif mod(floor(hp_fl_order),2) == 1
        hp_fl_order=floor(hp_fl_order)+1;
    end
    
    if mod(floor(lp_fl_order),2) == 0
        lp_fl_order=floor(lp_fl_order)+2;
    elseif mod(floor(lp_fl_order),2) == 1
        lp_fl_order=floor(lp_fl_order)+1;
    end
    
    % Calculate cutoff frequency
    high_cutoff = highpass/2;
    low_cutoff = lowpass + (low_transband/2);
    
    % Performing high pass filtering
    EEG = eeg_checkset( EEG );
    EEG = pop_firws(EEG, 'fcutoff', high_cutoff, 'ftype', 'highpass',...
        'wtype', 'hamming', 'forder', hp_fl_order, 'minphase', 0);
    EEG = eeg_checkset( EEG );
    
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    
    % pop_firws() - filter window type hamming ('wtype', 'hamming')
    % pop_firws() - applying zero-phase (non-causal) filter ('minphase', 0)
    
    % Performing low pass filtering
    EEG = eeg_checkset( EEG );
    EEG = pop_firws(EEG, 'fcutoff', low_cutoff, 'ftype', 'lowpass',...
        'wtype', 'hamming', 'forder', lp_fl_order, 'minphase', 0);
    EEG = eeg_checkset( EEG );
    
    % pop_firws() - transition band width: 10 Hz
    % pop_firws() - filter window type hamming ('wtype', 'hamming')
    % pop_firws() - applying zero-phase (non-causal) filter ('minphase', 0)
    
    % Plot PSD
    fig=compute_and_plot_psd(EEG,1:EEG.nbchan);
    saveas(fig, fullfile(subject_output_data_dir,'05-filtered_psd.png'));
    
    %% segment data using fixation intervals (Look Times) or bad intervals known apriori
    if(isSegt)
        if(~isempty(params.sname) && ~isempty(params.sloc))
            try
                lookFile=importdata([params.sloc filesep params.sname]);
            catch
                error('An error occurred in importing the segmentation file. If you think this is a bug, please report on the github repo issues section');
            end
        end
        
        if(~isempty(lookFile))
            try
                sheetName = name;
                lookTimes=NEAR_getLookTimes(lookFile,sheetName,look_thr);
            catch
                error('An error occurred in segmentation. Please find our template document in the repository to edit your time intervals.\n');
            end
        else
            error('We cannot find the file. Please check the file path and run again.\n');
        end
        % segment EEG data
        EEG = pop_select( EEG,'time',lookTimes);
        fprintf('\nSegmentation is done\n');
    end
    
    %% NEAR Bad Channel Detection
    if (isBadCh)
        
        [EEG, flat_ch, lof_ch, periodo_ch, LOF_vec, thresh_lof_update] = NEAR_getBadChannels(EEG, isFlat, flatWin, isLOF, thresh_lof, dist_metric, isAdapt, ...
            isPeriodogram, frange, winsize, winov, pthresh, isVisIns);
        save(fullfile(subject_output_data_dir, 'LOF_Values.mat'), 'LOF_vec'); % save .mat format
        disp('Bad Channel Detection is performed successfully');
        badChans = sort(unique(union(union(flat_ch, lof_ch),periodo_ch)));
        
        if(~isempty(badChans))
            if(size(badChans,1) ~= 1)
                badChans = badChans';
            end
        end
        
        
        if(isVisIns)
            % visual inspection and reject using 'Reject' button on the GUI
            colors = repmat({'k'},1, EEG.nbchan);
            
            for i = 1:length(periodo_ch)
                colors(1,periodo_ch(i)) = 	{[0.9290, 0.6940, 0.1250]};
            end
            
            for i = 1:length(lof_ch)
                colors(1,lof_ch(i)) = {'r'};
            end
            
            for i = 1:length(flat_ch)
                colors(1,flat_ch(i)) = {'r'};
            end
            
            eeg_rejmacro; % script macro for generating command and old rejection arrays
            
            eegplot(EEG.data, 'srate', EEG.srate, 'title', 'NEAR Bad Channels Plot (Red and Yellow Electrodes are bad)', ...
                'limits', [EEG.xmin EEG.xmax]*1000, 'color', colors,  'dispchans', 5, 'spacing', 500, eegplotoptions{:});
        end
        EEG = pop_select(EEG, 'nochannel', badChans);
        
        lof_flat_channels{s_idx}='';
        if numel(flat_ch)>0
            lof_flat_channels(s_idx)=join(cellfun(@(x) num2str(x(1)), num2cell(flat_ch,3), 'UniformOutput', false)',',');
        end
        lof_channels{s_idx}='';
        if numel(lof_ch)>0
            lof_channels(s_idx)=join(cellfun(@(x) num2str(x(1)), num2cell(lof_ch,3), 'UniformOutput', false)',',');
        end
        lof_periodo_channels{s_idx}='';
        if numel(periodo_ch)>0
            lof_periodo_channels(s_idx)=join(cellfun(@(x) num2str(x(1)), num2cell(periodo_ch,3), 'UniformOutput', false)',',');
        end
        lof_bad_channels{s_idx}='';
        if numel(badChans)>0
            lof_bad_channels(s_idx)=join(cellfun(@(x) num2str(x(1)), num2cell(badChans,3), 'UniformOutput', false)',',');
        end
    else
        disp('NEAR Bad Channel Detection is not employed. Set the variable ''isBadCh'' to 1 to enable bad channel detection');
    end
    
    %% Save data after running filter and LOF function, if saving interim results was preferred
    if save_interim_result ==1
        if output_format==1
            EEG = eeg_checkset( EEG );
            EEG = pop_editset(EEG, 'setname', strrep(data_file_name, ext, '_filtered_data'));
            EEG = pop_saveset( EEG,'filename',strrep(data_file_name, ext, '_filtered_data.set'),...
                'filepath', [subject_output_data_dir filesep '01_filtered_data' filesep]); % save .set format
        elseif output_format==2
            save([[subject_output_data_dir filesep '01_filtered_data' filesep ] strrep(data_file_name, ext, '_filtered_data.mat')], 'EEG'); % save .mat format
        end
    end
    
    fig=figure();
    topoplot([],EEG.chanlocs, 'style', 'blank',  'electrodes', 'labelpoint', 'chaninfo', EEG.chaninfo);
    saveas(fig, fullfile(subject_output_data_dir,'06-lof_removed.png'));
    
    %% Bad epochs correction/removal using ASR
    if(isBadSeg)
        EEG_copy = EEG;
        EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion','off','ChannelCriterion','off','LineNoiseCriterion','off', ...
            'Highpass','off','BurstCriterion',rej_cutoff,'WindowCriterion',add_reject,'BurstRejection',rej_mode,'Distance','Euclidian');
        
        if(strcmp(rej_mode, 'on'))
            modified_mask = ~EEG.etc.clean_sample_mask;
        else
            modified_mask = sum(abs(EEG_copy.data-EEG.data),1) > 1e-10;
        end
        
        tot_samples_modified = (length(find(modified_mask)) * 100) / EEG_copy.pnts;
        tot_samples_modified = round(tot_samples_modified * 100) / 100;
        asr_tot_samples_modified(s_idx)=tot_samples_modified;
        change_in_RMS = -(mean(rms(EEG.data,2)) - mean(rms(EEG_copy.data,2))*100)/mean(rms(EEG_copy.data,2)); % in percentage
        change_in_RMS = round(change_in_RMS * 100) / 100;
        asr_change_in_RMS(s_idx) =change_in_RMS;
        if(isVisIns)
            try
                vis_artifacts(EEG,EEG_copy);
            catch
                warning('vis_artifacts failed. Skipping visualization.')
            end
        end
        fprintf('\nArtifacted epochs are corrected by ASR algorithm\n');
    end
    
    %% Save data after running ASR function, if saving interim results was preferred
    if save_interim_result ==1
        if output_format==1
            EEG = eeg_checkset( EEG );
            EEG = pop_editset(EEG, 'setname', strrep(data_file_name, ext, '_asr_data'));
            EEG = pop_saveset( EEG,'filename',strrep(data_file_name, ext, '_asr_data.set'),...
                'filepath', [subject_output_data_dir filesep '02_near_data' filesep]); % save .set format
        elseif output_format==2
            save([[subject_output_data_dir filesep '02_near_data' filesep ] strrep(data_file_name, ext, '_asr_data.mat')], 'EEG'); % save .mat format
        end
    end
    
    fig=compute_and_plot_psd(EEG,1:EEG.nbchan);
    saveas(fig, fullfile(subject_output_data_dir,'07-asr_psd.png'));
    
    %% STEP 8: Prepare data for ICA
    EEG_copy=EEG; % make a copy of the dataset
    EEG_copy = eeg_checkset(EEG_copy);
    
    % Perform 1Hz high pass filter on copied dataset
    transband = 1;
    fl_cutoff = transband/2;
    fl_order = 3.3 / (transband / EEG.srate);
    
    if mod(floor(fl_order),2) == 0
        fl_order=floor(fl_order);
    elseif mod(floor(fl_order),2) == 1
        fl_order=floor(fl_order)+1;
    end
    
    EEG_copy = pop_firws(EEG_copy, 'fcutoff', fl_cutoff,...
        'ftype', 'highpass', 'wtype', 'hamming', 'forder', fl_order,...
        'minphase', 0);
    EEG_copy = eeg_checkset(EEG_copy);
    
    % Create 1 second epoch
    % insert temporary marker 1 second apart and create epochs
    EEG_copy=eeg_regepochs(EEG_copy,'recurrence', 1, 'limits',[0 1],...
        'rmbase', [NaN], 'eventtype', '999'); 
    EEG_copy = eeg_checkset(EEG_copy);
    
    % Find bad epochs and delete them from dataset
    % [lower upper] threshold limit(s) in mV.
    vol_thrs = [-1000 1000]; 
    % [lower upper] threshold limit(s) in dB.
    %emg_thrs = [-100 30]; 
    % [lower upper] frequency limit(s) in Hz.
    %emg_freqs_limit = [20 40]; 
    
    % Find channel/s with xx% of artifacted 1-second epochs and delete them
    chanCounter = 1; ica_prep_badChans = [];
    numEpochs =EEG_copy.trials; % find the number of epochs
    all_bad_channels=0;
    
    for ch=1:EEG_copy.nbchan
        % Find artifaceted epochs by detecting outlier voltage
        EEG_copy = pop_eegthresh(EEG_copy,1, ch, vol_thrs(1), vol_thrs(2),...
            EEG_copy.xmin, EEG_copy.xmax, 0, 0);
        EEG_copy = eeg_checkset( EEG_copy );
        
        % 1         : data type (1: electrode, 0: component)
        % 0         : display with previously marked rejections? (0: no, 1: yes)
        % 0         : reject marked trials? (0: no (but store the  marks), 1:yes)
        
        % Find artifaceted epochs by using thresholding of frequencies in the data.
        % this method mainly rejects muscle movement (EMG) artifacts
%         EEG_copy = pop_rejspec( EEG_copy, 1,'elecrange',ch ,'method','fft',...
%             'threshold', emg_thrs, 'freqlimits', emg_freqs_limit,...
%             'eegplotplotallrej', 0, 'eegplotreject', 0);
        
        % method                : method to compute spectrum (fft)
        % threshold             : [lower upper] threshold limit(s) in dB.
        % freqlimits            : [lower upper] frequency limit(s) in Hz.
        % eegplotplotallrej     : 0 = Do not superpose rejection marks on previous marks stored in the dataset.
        % eegplotreject         : 0 = Do not reject marked trials (but store the  marks).
        
        % Find number of artifacted epochs
        EEG_copy = eeg_checkset( EEG_copy );
        EEG_copy = eeg_rejsuperpose( EEG_copy, 1, 1, 1, 1, 1, 1, 1, 1);
        artifacted_epochs=EEG_copy.reject.rejglobal;
        
        % Find bad channel / channel with more than 20% artifacted epochs
        if sum(artifacted_epochs) > (numEpochs*20/100)
            ica_prep_badChans(chanCounter) = ch;
            chanCounter=chanCounter+1;
        end
    end
    
    % If all channels are bad, save the dataset at this stage and ignore the remaining of the preprocessing.
    if numel(ica_prep_badChans)==EEG.nbchan || numel(ica_prep_badChans)+1==EEG.nbchan
        all_bad_channels=1;
        warning(['No usable data for datafile', data_file_name]);        
    else
        % Reject bad channel - channel with more than xx% artifacted epochs
        EEG_copy = pop_select( EEG_copy,'nochannel', ica_prep_badChans);
        EEG_copy = eeg_checkset(EEG_copy);
    end
    
    if numel(ica_prep_badChans)==0
        all_ica_preparation_bad_channels{s_idx}='0';
    else
        all_ica_preparation_bad_channels{s_idx}=num2str(ica_prep_badChans);
    end
    
    if all_bad_channels == 1
        length_ica_data(s_idx)=0;
        total_ICs(s_idx)=0;
        ICs_removed{s_idx}='0';
        total_epochs_before_artifact_rejection(s_idx)=0;
        total_epochs_after_artifact_rejection(s_idx)=0;
        total_channels_interpolated(s_idx)=0;
        continue % ignore rest of the processing and go to next datafile
    end
    
    % Find the artifacted epochs across all channels and reject them before doing ICA.
    EEG_copy = pop_eegthresh(EEG_copy,1, 1:EEG_copy.nbchan, vol_thrs(1),...
        vol_thrs(2), EEG_copy.xmin, EEG_copy.xmax,0,0);
    EEG_copy = eeg_checkset(EEG_copy);
    
    % 1         : data type (1: electrode, 0: component)
    % 0         : display with previously marked rejections? (0: no, 1: yes)
    % 0         : reject marked trials? (0: no (but store the  marks), 1:yes)
    
    % Find artifaceted epochs by using power threshold in 20-40Hz frequency band.
    % This method mainly rejects muscle movement (EMG) artifacts.
%     EEG_copy = pop_rejspec(EEG_copy, 1,'elecrange', 1:EEG_copy.nbchan,...
%         'method', 'fft', 'threshold', emg_thrs ,'freqlimits', emg_freqs_limit,...
%         'eegplotplotallrej', 0, 'eegplotreject', 0);
    
    % method                : method to compute spectrum (fft)
    % threshold             : [lower upper] threshold limit(s) in dB.
    % freqlimits            : [lower upper] frequency limit(s) in Hz.
    % eegplotplotallrej     : 0 = Do not superpose rejection marks on previous marks stored in the dataset.
    % eegplotreject         : 0 = Do not reject marked trials (but store the  marks).
    
    % Find the number of artifacted epochs and reject them
    EEG_copy = eeg_checkset(EEG_copy);
    EEG_copy = eeg_rejsuperpose(EEG_copy, 1, 1, 1, 1, 1, 1, 1, 1);
    reject_artifacted_epochs=EEG_copy.reject.rejglobal;
    EEG_copy = pop_rejepoch(EEG_copy, reject_artifacted_epochs, 0);
    
    fig=compute_and_plot_psd(EEG_copy, 1:EEG_copy.nbchan);
    saveas(fig, fullfile(subject_output_data_dir,'08-ica_copy_epochs_psd.png'));    
    
    %% STEP 9: Run ICA
    length_ica_data(s_idx)=EEG_copy.trials; % length of data (in second) fed into ICA
    EEG_copy = eeg_checkset(EEG_copy);
    EEG_copy = pop_runica(EEG_copy, 'icatype', 'runica', 'extended', 1,...
        'stop', 1E-7, 'interupt','off');
    
    if save_interim_result==1
        if output_format==1
            EEG_copy = eeg_checkset(EEG_copy);
            EEG_copy = pop_editset(EEG_copy, 'setname',  strrep(data_file_name, ext, '_ica'));
            EEG_copy = pop_saveset(EEG_copy, 'filename', strrep(data_file_name, ext, '_ica.set'),...
                'filepath', [subject_output_data_dir filesep '03_ica_data' filesep ]); % save .set format
        elseif output_format==2
            save([[subject_output_data_dir filesep '03_ica_data' filesep ] strrep(data_file_name, ext, '_ica.mat')], 'EEG_copy'); % save .mat format
        end
    end
    
    % Find the ICA weights that would be transferred to the original dataset
    ICA_WINV=EEG_copy.icawinv;
    ICA_SPHERE=EEG_copy.icasphere;
    ICA_WEIGHTS=EEG_copy.icaweights;
    ICA_CHANSIND=EEG_copy.icachansind;
    
    % If channels were removed from copied dataset during preparation of ica, then remove
    % those channels from original dataset as well before transferring ica weights.
    EEG = eeg_checkset(EEG);
    EEG = pop_select(EEG,'nochannel', ica_prep_badChans);
    
    % Transfer the ICA weights of the copied dataset to the original dataset
    EEG.icawinv=ICA_WINV;
    EEG.icasphere=ICA_SPHERE;
    EEG.icaweights=ICA_WEIGHTS;
    EEG.icachansind=ICA_CHANSIND;
    EEG = eeg_checkset(EEG);
    
    %% STEP 10: Run adjust to find artifacted ICA components
    badICs=[];
    
    if size(EEG_copy.icaweights,1) == size(EEG_copy.icaweights,2)
        figure()
        badICs = adjusted_ADJUST(EEG_copy, [[subject_output_data_dir filesep '03_ica_data' filesep] strrep(data_file_name, ext, '_adjust_report')]);        
        close all;
    else % if rank is less than the number of electrodes, throw a warning message
        warning('The rank is less than the number of electrodes. ADJUST will be skipped. Artefacted ICs will have to be manually rejected for this participant');
    end
    
    % Mark the bad ICs found by ADJUST
    for ic=1:length(badICs)
        EEG.reject.gcompreject(1, badICs(ic))=1;
        EEG = eeg_checkset(EEG);
    end
    total_ICs(s_idx)=size(EEG.icasphere, 1);
    if numel(badICs)==0
        ICs_removed{s_idx}='0';
    else
        ICs_removed{s_idx}=num2str(double(badICs));
    end
    % Mark the bad 1/f found by adjust
    % if numel(cwb)==0
    %    bad_1_f{s_idx}='0';
    %else
    %   bad_1_f{s_idx}=num2str(double(cwb));
    %end
    
    
    %% Save dataset after ICA, if saving interim results was preferred
    if save_interim_result==1
        if output_format==1
            EEG = eeg_checkset(EEG);
            EEG = pop_editset(EEG, 'setname',  strrep(data_file_name, ext, '_ica_data'));
            EEG = pop_saveset(EEG, 'filename', strrep(data_file_name, ext, '_ica_data.set'),...
                'filepath', [subject_output_data_dir filesep '03_ica_data' filesep ]); % save .set format
        elseif output_format==2
            save([[subject_output_data_dir filesep '03_ica_data' filesep ] strrep(data_file_name, ext, '_ica_data.mat')], 'EEG'); % save .mat format
        end
    end
    
    %% STEP 11: Remove artifacted ICA components from data
    all_bad_ICs=0;
    ICs2remove=find(EEG.reject.gcompreject); % find ICs to remove
    
    % If all ICs and bad, save data at this stage and ignore rest of the preprocessing for this subject.
    if numel(ICs2remove)==total_ICs(s_idx)
        all_bad_ICs=1;
        warning(['No usable data for datafile', data_file_name]);        
    else
        EEG = eeg_checkset( EEG );
        EEG = pop_subcomp( EEG, ICs2remove, 0); % remove ICs from dataset
    end
    
    if all_bad_ICs==1
        total_epochs_before_artifact_rejection(s_idx)=0;
        total_epochs_after_artifact_rejection(s_idx)=0;
        total_channels_interpolated(s_idx)=0;
        continue % ignore rest of the processing and go to next datafile
    end
    
    % Notch filter
    %EEG = pop_eegfiltnew(EEG, 59, 61, 1650, 1);
    
    fig=compute_and_plot_psd(EEG, 1:EEG.nbchan);
    saveas(fig, fullfile(subject_output_data_dir,'09-ica_art_rej_psd.png'));
    
    
    %% STEP 12: Segment data into fixed length epochs
    if epoch_data==1
        if task_eeg ==1 % task eeg
            EEG = eeg_checkset(EEG);
            EEG = pop_epoch(EEG, task_event_markers, task_epoch_length, 'epochinfo', 'yes');
        elseif task_eeg==0 % resting eeg
            if overlap_epoch==1
                EEG=eeg_regepochs(EEG,'recurrence',(rest_epoch_length/2),'limits',[0 rest_epoch_length], 'rmbase', [NaN], 'eventtype', char(dummy_events));
                EEG = eeg_checkset(EEG);
            else
                EEG=eeg_regepochs(EEG,'recurrence',rest_epoch_length,'limits',[0 rest_epoch_length], 'rmbase', [NaN], 'eventtype', char(dummy_events));
                EEG = eeg_checkset(EEG);
            end
        end
    end
    fig=compute_and_plot_psd(EEG, 1:EEG.nbchan);
    saveas(fig, fullfile(subject_output_data_dir,'10-epoch_psd.png'));
    
    total_epochs_before_artifact_rejection(s_idx)=EEG.trials;
    
    %% STEP 13: Remove baseline
    if remove_baseline==1
        EEG = eeg_checkset( EEG );
        EEG = pop_rmbase( EEG, baseline_window);
    end
    
    %% Step 14: Artifact rejection
    all_bad_epochs=0;
    if voltthres_rejection==1 % check voltage threshold rejection
        if interp_epoch==1 % check epoch level channel interpolation
            chans=[]; chansidx=[];chans_labels2=[];
            chans_labels2=cell(1,EEG.nbchan);
            for i=1:EEG.nbchan
                chans_labels2{i}= EEG.chanlocs(i).labels;
            end
            [chans,chansidx] = ismember(frontal_channels, chans_labels2);
            frontal_channels_idx = chansidx(chansidx ~= 0);
            badChans = zeros(EEG.nbchan, EEG.trials);
            badepoch=zeros(1, EEG.trials);
            if isempty(frontal_channels_idx)==1 % check whether there is any frontal channel in dataset to check
                warning('No frontal channels from the list present in the data. Only epoch interpolation will be performed.');
            else
                % find artifaceted epochs by detecting outlier voltage in the specified channels list and remove epoch if artifacted in those channels
                for ch =1:length(frontal_channels_idx)
                    EEG = pop_eegthresh(EEG,1, frontal_channels_idx(ch), volt_threshold(1), volt_threshold(2), EEG.xmin, EEG.xmax,0,0);
                    EEG = eeg_checkset( EEG );
                    EEG = eeg_rejsuperpose( EEG, 1, 1, 1, 1, 1, 1, 1, 1);
                    badChans(ch,:) = EEG.reject.rejglobal;
                end
                for ii=1:size(badChans, 2)
                    badepoch(ii)=sum(badChans(:,ii));
                end
                badepoch=logical(badepoch);
            end
            
            % If all epochs are artifacted, save the dataset and ignore rest of the preprocessing for this subject.
            if sum(badepoch)==EEG.trials || sum(badepoch)+1==EEG.trials
                all_bad_epochs=1;
                warning(['No usable data for datafile', data_file_name]);                
            else
                EEG = pop_rejepoch( EEG, badepoch, 0);
                EEG = eeg_checkset(EEG);
            end
            
            if all_bad_epochs==0
                % Interpolate artifacted data for all reaming channels
                badChans = zeros(EEG.nbchan, EEG.trials);
                % Find artifacted epochs by detecting outlier voltage but don't remove
                for ch=1:EEG.nbchan
                    EEG = pop_eegthresh(EEG,1, ch, volt_threshold(1), volt_threshold(2), EEG.xmin, EEG.xmax,0,0);
                    EEG = eeg_checkset(EEG);
                    EEG = eeg_rejsuperpose(EEG, 1, 1, 1, 1, 1, 1, 1, 1);
                    badChans(ch,:) = EEG.reject.rejglobal;
                end
                tmpData = zeros(EEG.nbchan, EEG.pnts, EEG.trials);
                for e = 1:EEG.trials
                    % Initialize variables EEGe and EEGe_interp;
                    EEGe = []; EEGe_interp = []; badChanNum = [];
                    % Select only this epoch (e)
                    EEGe = pop_selectevent( EEG, 'epoch', e, 'deleteevents', 'off', 'deleteepochs', 'on', 'invertepochs', 'off');
                    badChanNum = find(badChans(:,e)==1); % find which channels are bad for this epoch
                    if length(badChanNum) < round((10/100)*EEG.nbchan)% check if more than 10% are bad
                        EEGe_interp = eeg_interp(EEGe,badChanNum); %interpolate the bad channels for this epoch
                        tmpData(:,:,e) = EEGe_interp.data; % store interpolated data into matrix
                    end
                end
                EEG.data = tmpData; % now that all of the epochs have been interpolated, write the data back to the main file
                
                % If more than 10% of channels in an epoch were interpolated, reject that epoch
                badepoch=zeros(1, EEG.trials);
                for ei=1:EEG.trials
                    NumbadChan = badChans(:,ei); % find how many channels are bad in an epoch
                    if sum(NumbadChan) > round((10/100)*EEG.nbchan)% check if more than 10% are bad
                        badepoch (ei)= sum(NumbadChan);
                    end
                end
                badepoch=logical(badepoch);
            end
            % If all epochs are artifacted, save the dataset and ignore rest of the preprocessing for this subject.
            if sum(badepoch)==EEG.trials || sum(badepoch)+1==EEG.trials
                all_bad_epochs=1;
                warning(['No usable data for datafile', data_file_name]);                
            else
                EEG = pop_rejepoch(EEG, badepoch, 0);
                EEG = eeg_checkset(EEG);
            end
        else % if no epoch level channel interpolation
            EEG = pop_eegthresh(EEG, 1, (1:EEG.nbchan), volt_threshold(1), volt_threshold(2), EEG.xmin, EEG.xmax, 0, 0);
            EEG = eeg_checkset(EEG);
            EEG = eeg_rejsuperpose( EEG, 1, 1, 1, 1, 1, 1, 1, 1);
        end % end of epoch level channel interpolation if statement
        
        % If all epochs are artifacted, save the dataset and ignore rest of the preprocessing for this subject.
        if sum(EEG.reject.rejthresh)==EEG.trials || sum(EEG.reject.rejthresh)+1==EEG.trials
            all_bad_epochs=1;
            warning(['No usable data for datafile', data_file_name]);            
        else
            EEG = pop_rejepoch(EEG,(EEG.reject.rejthresh), 0);
            EEG = eeg_checkset(EEG);
        end
    end % end of voltage threshold rejection if statement
    
    % if all epochs are found bad during artifact rejection
    if all_bad_epochs==1
        total_epochs_after_artifact_rejection(s_idx)=0;
        total_channels_interpolated(s_idx)=0;
        continue % ignore rest of the processing and go to next datafile
    else
        total_epochs_after_artifact_rejection(s_idx)=EEG.trials;        
    end
    
    %% Interpolation    
    if(isInterp)
        EEG = pop_interp(EEG, origEEG.chanlocs, interp_type);
        fprintf('\nMissed channels are spherically interpolated\n');
    end
    if numel(badChans)==0 && numel(ica_prep_badChans)==0
        total_channels_interpolated(s_idx)=0;
    else
        total_channels_interpolated(s_idx)=numel(badChans)+ numel(ica_prep_badChans);
    end
    
    %% Re-referencing
    if(isempty(reref))
        warning('Skipping rereferencing as the parameter reref is empty. An example setup: reref = {''Cz''} or reref = [30]');
    else
        if(isAvg) % average referencing
            
            if(isnumeric(reref))
                EEG = pop_chanedit(EEG, 'setref',{1:EEG.nbchan, reref});
            else
                labels = {EEG.chanlocs.labels};
                ch_idx = find(ismember(labels, reref)); %optimized code
                if(isempty(ch_idx)); warning('The reference channel label(s) does not exist in the dataset. Please check the channel locations file.');end
                EEG = pop_chanedit(EEG, 'setref',{1:EEG.nbchan, ch_idx});
            end
            EEG = pop_reref( EEG, []);
            
        else % otherwise
            
            if(isnumeric(reref))
                EEG = pop_reref( EEG, reref);
            else
                labels = {EEG.chanlocs.labels};
                ch_idx = find(ismember(labels, reref)); %optimized code for multi-labelled cell string array
                if(isempty(ch_idx)); warning('The reference channel label(s) does not exist in the dataset. Please check the channel locations file.');end
                EEG = pop_reref( EEG, ch_idx);
                refch = cell2mat(reref);
            end
            
        end
    end
    
    fig=compute_and_plot_psd(EEG, 1:EEG.nbchan);
    saveas(fig, fullfile(subject_output_data_dir,'11-art_rej_reref_psd.png'));    
    
    %% Save processed data
    if output_format==1
        EEG = eeg_checkset(EEG);
        EEG = pop_editset(EEG, 'setname',  strrep(data_file_name, ext, '_rereferenced_data'));
        EEG = pop_saveset(EEG, 'filename', strrep(data_file_name, ext, '_rereferenced_data.set'),...
            'filepath', [subject_output_data_dir filesep '04_rereferenced_data' filesep ]); % save .set format
    elseif output_format==2
        save([[subject_output_data_dir filesep '04_rereferenced_data' filesep ] strrep(data_file_name, ext, '_rereferenced_data.mat')], 'EEG'); % save .mat format
    end
    
end

disp(size(study_info.participant_info.participant_id))
disp(size(lof_flat_channels))
disp(size(lof_channels))
disp(size(lof_periodo_channels))
disp(size(lof_bad_channels))
disp(size(ars_tot_samples_modified))
disp(size(ars_change_in_RMS))
disp(size(ica_preparation_bad_channels))
disp(size(length_ica_data))
disp(size(total_ICs))
disp(size(ICs_removed))
disp(size(total_epochs_before_artifact_rejection))
disp(size(total_epochs_after_artifact_rejection))
disp(size(total_channels_interpolated))

%% Create the report table for all the data files with relevant preprocessing outputs.
report_table=table(study_info.participant_info.participant_id,...
    lof_flat_channels', lof_channels', lof_periodo_channels', lof_bad_channels',...
    asr_tot_samples_modified', asr_change_in_RMS', ica_preparation_bad_channels',...
    length_ica_data', total_ICs', ICs_removed', total_epochs_before_artifact_rejection',...
    total_epochs_after_artifact_rejection',total_channels_interpolated');

report_table.Properties.VariableNames={'subject','lof_flat_channels', 'lof_channels', ...
    'lof_periodo_channels', 'lof_bad_channels', 'asr_tot_samples_modified', 'asr_change_in_RMS',...
    'ica_preparation_bad_channels', 'length_ica_data', 'total_ICs', 'ICs_removed', 'total_epochs_before_artifact_rejection', ...
    'total_epochs_after_artifact_rejection', 'total_channels_interpolated'};
writetable(report_table, fullfile(study_info.data_dir, 'derivatives', 'NEARICA', ['NEARICA_preprocessing_report_', datestr(now,'dd-mm-yyyy'),'.csv']));
end