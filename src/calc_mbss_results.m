function calc_mbss_result()

clear all;
close all;
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
pooling             = 'max';      % Pooling method {'max' 'sum'}
normalizeSpecInst   = 0;          % Normalize instantaneous local angular spectra (1:normalization 0:no normalization)
c                   = 343;        % Speed of sound (m.s-1)
% Search space
thetaBound   = [-179 180];        % Azimuth (Theta) search boundaries (°)
phiBound     = [-90 90];          % Elevation (Phi) search boundaries (°)
thetaPhiRes  = 1;                 % Resolution (°) of the global 3D reference system {theta (azimuth),phi (elevation)}
alphaRes     = 5;                 % Resolution (°) of the 2D reference system defined for each microphone pair
% Multiple sources parameters
%nsrc         = 2;                 % Number of sources to be detected
minAngle     = 10;                % Minimum angle between peaks
% Display results
specDisplay  = 0;                 % Display angular spectrum found and results

tic
all_cat = {dir_tnoisy};
all_nsrc = [2];

for c = 1:numel(all_cat)
	catg = all_cat{c};
	fprintf('Category: %s.\n', catg);
	files = dir(fullfile(catg,'*.wav'));
	files = {files.name};
	for i = 1:numel(files)
		fname = files{i};
		path = fullfile(catg, fname);
		root_fname = fullfile(catg,fname(1:end-4));

		% Get geoArray from file name
		geo = regexp(fname,'(?<=arrayGeo)((\d)+)(?=_)','match');
		geo = str2num(geo{1});
		mic_pos = i2geo(geo);
		fprintf('Using geoArray %d.\n',geo);

		%% Wav read and resample to 16 kHz
		[x,fs] = audioread(path);
		%size(x)
		%x = x(1:8000,:);

		for nsrc = 1:numel(all_nsrc)
			dirs_fname = [root_fname, '_MBSS_nsrc', num2str(nsrc), '.txt'];
			%% Call Multi_channel_bss_locate_spec on x
			%tic();
			[theta_e, phi_e] = MBSS_locate_spec(x, fs, nsrc , mic_pos, c, angularSpectrumMeth, pooling, thetaBound, phiBound, thetaPhiRes, alphaRes, minAngle, normalizeSpecInst, specDisplay);
			%e = toc();
			%fprintf('time: %.2f\n',e);
			%
			%[theta_e,phi_e]
			%% Print the result
			%for i = 1:nsrc
				%fprintf('Estimated source %d : \n Azimuth (Theta): %.0f \t Elevation (Phi): %.0f \n\n',i,theta_e(i),phi_e(i));
			%end
		end
	end
end
total_time = toc

end

function mic_pos = i2geo(i)
	f =  fullfile('data','annotations','arrays',['arrayGeo',num2str(i),'.txt']);
	mic_pos = dlmread(f, '\t', [0, 1, 7, 3]);
end
