function res =  mbss_python_wrapper(fname, from_sec, to_sec, nsrc, pooling)
if nargin < 5
	pooling ='max'
end

theta_e = [];
phi_e = [];

addpath('src')

%% Localization tools to path
addpath(fullfile('src','third-party','MBSS_v1.3','mFiles','localization_tools'));
dir_noisy = fullfile('data', 'audio', 'noisy');
dir_noises = fullfile('data', 'audio', 'noises');
dir_tnoisy = fullfile('data', 'audio', 'noisy-tiny');
dir_tnoises = fullfile('data', 'audio', 'noises-tiny');

%fileName = 'male_female_mixture.wav';  % see readme.txt

%% Algorithm PARAMETERS
angularSpectrumMeth = 'GCC-PHAT'; % Local angular spectrum method {'GCC-PHAT' 'GCC-NONLIN' 'MVDR' 'MVDRW' 'DS' 'DSW' 'DNM' 'MUSIC'}
%pooling             = 'max';      % Pooling method {'max' 'sum'}
normalizeSpecInst   = 0;          % Normalize instantaneous local angular spectra (1:normalization 0:no normalization)
c                   = 343;        % Speed of sound (m.s-1)
% Search space
thetaBound   = [-179 180];        % Azimuth (Theta) search boundaries (°)
phiBound     = [-90 90];          % Elevation (Phi) search boundaries (°)
thetaPhiRes  = 2;                 % Resolution (°) of the global 3D reference system {theta (azimuth),phi (elevation)}
alphaRes     = 5;                 % Resolution (°) of the 2D reference system defined for each microphone pair
% Multiple sources parameters
%nsrc         = 2;                 % Number of sources to be detected
minAngle     = 10;                % Minimum angle between peaks
% Display results
specDisplay  = 0;                 % Display angular spectrum found and results

%tic
%all_cat = {dir_tnoisy};

root_fname = fname(1:end-4);

% Get geoArray from file name
geo = regexp(fname,'(?<=arrayGeo)((\d)+)(?=_)','match');
geo = str2num(geo{1});
mic_pos = i2geo(geo);
%fprintf('Using geoArray %d.\n',geo);

%% Wav read and resample to 16 kHz
[x,fs] = audioread(fname);
sx = size(x);
beg_sample = 1+floor(from_sec*fs);
end_sample = min(1+ceil(to_sec*fs),sx(1));
%size(x)
x = x(beg_sample:end_sample,:);
%x = x(1:8000,:);

%dirs_fname = [root_fname, '_MBSS_nsrc', num2str(nsrc), '.txt'];
%% Call Multi_channel_bss_locate_spec on x
%disp(nsrc)
[theta_e, phi_e] = MBSS_locate_spec(x, fs, nsrc , mic_pos, c, angularSpectrumMeth, pooling, thetaBound, phiBound, thetaPhiRes, alphaRes, minAngle, normalizeSpecInst, specDisplay);
%
%[theta_e,phi_e]
%% Print the result
%for i = 1:nsrc
	%fprintf('Estimated source %d : \n Azimuth (Theta): %.0f \t Elevation (Phi): %.0f \n\n',i,theta_e(i),phi_e(i));
%end
%disp(size(theta_e))
%total_time = toc
%
res = [theta_e', phi_e'];

end

function mic_pos = i2geo(i)
	f =  fullfile('data','annotations','arrays',['arrayGeo',num2str(i),'.txt']);
	mic_pos = dlmread(f, '\t', [0, 1, 7, 3]);
end
