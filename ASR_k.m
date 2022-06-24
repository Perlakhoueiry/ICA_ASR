

%% Clear variable space and run eeglab

addpath('NoiseTools');
addpath('C:\Users\khoueiry\Desktop\eeglab2021.1');
dloc= 'C:\Users\khoueiry\Desktop\ASR_k';
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

params.rej_cutoff = 21;   % A lower value implies severe removal (Recommended value range: 20 to 30)
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
total_pband=[];
total_resids_pxx=[];
total_all_psd=[];

%Split the subjects into trainind sets and test set
group=size(study_info.participant_info,1);
a=0.1;
c = cvpartition(group,'Holdout',a,'Stratify',false);
set = training(c);
num= test(c);
test_subjs= find (num);
training_subjs=find(set);

for t_idx=1:length(training_subjs)
    s_idx=training_subjs(t_idx);
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
    %saveas(fig, fullfile(subject_output_data_dir,'03-inner_ch_locations.png'));
    
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
    rawEEG = EEG;
    % Range of ASR parameters
    range_k = [10 50];
    k_step  = 1;
    burstRej = 'off';
    k_in_array = range_k(1):k_step:range_k(2);
       
    for p = 1:length(k_in_array)
 
        fname   = subject;
        tmpName = strsplit(fname, '%s_task-%s_eeg.set');
        % first cell of split contains the name of the dataset
        sname   = tmpName{1};
        beta_band= [14 17];
        EEG1= pop_clean_rawdata(EEG, 'FlatlineCriterion','off',...
            'ChannelCriterion','off','LineNoiseCriterion','off',...
            'Highpass','off','BurstCriterion',k_in_array(p),...
            'WindowCriterion','off','BurstRejection','off','Distance','Euclidian');
    
        data=EEG1.data;
        winsize=EEG1.srate;
        overlap=round(winsize/2);
        [pxx,frex,~,~,~] = spectopo(EEG1.data,EEG1.pnts,...
            EEG1.srate, 'winsize', winsize, 'overlap', overlap,...
            'plot', 'off', 'freqfac',2);
        
        % Find c3 and C4 cluster channels
        c3_idx=find(strcmp(study_info.clusters,'C3'));
        c3_channels=study_info.cluster_channels{c3_idx};
        c4_idx=find(strcmp(study_info.clusters,'C4'));
        c4_channels=study_info.cluster_channels{c4_idx};
        all_channels=c3_channels;
        all_channels(1,end+1:end+length(c4_channels))=c4_channels;
        
        chan_idx=[];
        for cidx=1:length(all_channels)
            idx=find(strcmp({EEG.chanlocs.labels},all_channels{cidx}));
            if length(idx)
                chan_idx(end+1)=idx;
            end
        end
%         cellfun(@(x) find(strcmp({EEG.chanlocs.labels},x)),...
%             all_channels);

        % Chop off 2Hz and > 100Hz from frex and all_psd
        freq_idx=find((frex>=10) & (frex<20));
        
        % Average over cluster channels
        all_psd=mean(pxx(chan_idx,freq_idx));
        total_all_psd(p,t_idx,:)=all_psd;
        
        % Find idx for beta
        beta_idx=knnsearch(frex(freq_idx),beta_band');
        
        % Fit 1/f to spectrum
        oof=log10(1./frex(freq_idx));
        lm_psd=fitlm(oof,all_psd,'RobustOpts','on');
        % Get fitted 1/f function
        fitted=lm_psd.Fitted;
        resids_pxx=lm_psd.Residuals.Raw;
        %resids_pxx=resids_pxx-min(resids_pxx);
        total_resids_pxx(p,t_idx,:)=resids_pxx;
        %compute the beta power
        pband=sum(resids_pxx(beta_idx(1):beta_idx(2)));
        total_pband(t_idx,p)=pband;

    end   
  
    
end
% figure()
% plot(frex(freq_idx),total_resids_pxx(1,:))
% title('k=10')
% figure()
% plot (frex(freq_idx),total_resids_pxx(21,:))
% title('k=30')
% figure()
% plot(frex(freq_idx),total_resids_pxx(41,:))
% title('k=50')
% figure()
% hold all
% plot(frex(freq_idx),total_resids_pxx(1,:),'g',frex(freq_idx),total_resids_pxx(21,:),'b',frex(freq_idx),total_resids_pxx(41,:),'r')
% legend('k=10','k=30','k=50')
save('ASR_calibration_9-month-old.mat', 'total_resids_pxx','total_pband','k_in_array')
    
    %% Figure
   
figure();
shadedErrorBar(k_in_array, mean(total_pband,1), std(total_pband)/sqrt(size(total_pband,1)),'lineProps','-b','transparent',1);
xlabel('k');
ylabel('beta power');

figure()
hold all
plot(k_in_array,total_pband);
xlabel('k');
ylabel('beta power');

