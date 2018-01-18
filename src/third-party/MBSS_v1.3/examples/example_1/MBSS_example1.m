% File MBSS_example1.m
%
% Apply multi-channel BSS Locate algorithm to male_female_mixture.wav multi-channel audio file
%
% Version : v1.0
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2015 Ewen Camberlein and Romain Lebarbenchon
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
% If you find it useful, please cite the following reference:
% http://bass-db.gforge.inria.fr/bss_locate/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;

%% Localization tools to path
addpath('./../../mFiles/localization_tools/');
addpath('./wav files');

%% Input File & Mic config
% Input wav file recorded by sensor corresponding to above locations
fileName = 'male_female_mixture.wav';  % see readme.txt

% Cartesian coordinates of the microphones (meters)
micPos = ... % All microphones are supposed to be omnidirectionnals !
         ...x      y       z
        [2.742	2.582	1.155;  %mic 1
	     2.671	2.582	1.231;  %mic 2
	     2.649	2.563	1.155;  %mic 3
	     2.649	2.492	1.231;  %mic 4
	     2.668	2.470	1.155;  %mic 5
	     2.739	2.470	1.231;  %mic 6
	     2.761	2.489	1.155;  %mic 7
	     2.761	2.560	1.231]; %mic 8

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
nsrc         = 2;                 % Number of sources to be detected
minAngle     = 10;                % Minimum angle between peaks
% Display results
specDisplay  = 1;                 % Display angular spectrum found and results

%% Wav read and resample to 16 kHz
[x,fs] = audioread(fileName);

%% Call Multi_channel_bss_locate_spec on x
[theta_e, phi_e] = MBSS_locate_spec(x, fs, nsrc , micPos, c, angularSpectrumMeth, pooling, thetaBound, phiBound, thetaPhiRes, alphaRes, minAngle, normalizeSpecInst, specDisplay);

%% Print the result
for i = 1:nsrc
    fprintf('Estimated source %d : \n Azimuth (Theta): %.0f \t Elevation (Phi): %.0f \n\n',i,theta_e(i),phi_e(i));
end