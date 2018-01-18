% File MBSS_example2.m
%
% Script example to  :
%    - Compute a simulated mixture of two static sources recorded by a
%    microphone array of 8 microphones with the help of Roomsimove toolbox ;
%    - Apply multi-channel BSS Locate algorithm on the mixture to estimate sources directions;
%    - Evaluate localization results between estimated angle and ground
%    truth (Recall, Precision, F-measure, Accuracy).
%
% Version : v1.3
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

%% Add tools to path
addpath('./../../mFiles/localization_tools/');
addpath('./../../mFiles/evaluation_tools/');
addpath('./../../mFiles/roomsimove_tools/');
addpath('./wav files/');

%% 16 kHz Mixture generation Parameters
fs = 16000;     % Sampling frequency for localisation processing (expert parameter : do not modify !)
[roomStruct,sensorsStruct,sourcesStruct] = MBSS_audioScene_parameters();
% Number of Sources
nsrcMixture = length(sourcesStruct);

%% Localization Algorithm Parameters
angularSpectrumMeth = 'GCC-PHAT';    % Local Angular spectrum method {'GCC-PHAT' 'GCC-NONLIN' 'MVDR' 'MVDRW' 'DS' 'DSW' 'DNM' 'MUSIC'}
pooling             = 'max';         % Pooling method {'max' 'sum'}
normalizeSpecInst   = 0;             % Normalize instantaneous local angular spectra (1:normalization 0:no normalization)
c                   = 343;           % Speed of sound (m.s-1)
% Search space
thetaBound       = [-179 180];       % Azimuth (Theta) search boundaries (°)
phiBound         = [-90 90];         % Elevation (Phi) search boundaries (°)
thetaPhiRes      = 1;                % Resolution (°) of the global 3D reference system {theta (azimuth),phi (elevation)}
alphaRes         = 5;                % Resolution (°) of the 2D reference system defined for each microphone pair
% Multiple sources parameters
nsrcLocalization = nsrcMixture;      % Number of sources to be detected
minAngle         = 10;               % Minimum angle between peaks
% Display results
specDisplay      = 1;                % Display angular spectrum found and sources directions found

%% Evaluation Parameters
angleThreshold = 10;                 % Maximum error between estimated and reference angle for results evaluation (Recall, Precision, F-measure)

%% Generate the simulated mixture with roomsimove
fprintf('Mixture generation (%d sources)\n',nsrcMixture);
% Call roomsimove toolbox
simg = []; % Sources images

for i = 1:nsrcMixture
    fprintf(' - Generation of source %d / %d image\n',i,nsrcMixture);
    
    [s,fsFile] = audioread(sourcesStruct(i).filename);
    % Downsampling signal to 16kHz if necessary
    if (fsFile > fs)
        [q,p] = rat(fs/fsFile);
        s = resample(s,q,p);
    elseif (fsFile < fs)
        error('[MBSS_example2.m error] wav file sampling frequency is below 16kHz : Upsampling the signal is not allowed');
    end
    % Generate room filter
    [time,HH] = MBSS_roomsimove(fs,roomStruct.room_size,roomStruct.F_abs',roomStruct.A',sensorsStruct.sensor_xyz',sensorsStruct.sensor_off',sensorsStruct.sensor_type,sourcesStruct(i).ptime,sourcesStruct(i).source_xyz);
    % Compute source image
    simg = cat(3,simg,MBSS_roomsimove_apply(time,HH,s',fs));
end

% Generate mixture : mean of source images
x = squeeze(mean(simg,3));

%% Call Multi_channel_bss_locate_spec on x
fprintf('\nLocalization processing\n');
[theta_e, phi_e] = MBSS_locate_spec(x', fs, nsrcLocalization , sensorsStruct.sensor_xyz, c, angularSpectrumMeth, pooling, thetaBound, phiBound, thetaPhiRes, alphaRes, minAngle, normalizeSpecInst, specDisplay);

% diplay sources found
fprintf('\nResults:\n',i);
for i = 1:nsrcLocalization
    fprintf(' - Estimated source %d:\n',i);
    fprintf('   Azimuth (Theta): %.2f°\t Elevation (Phi): %.2f°\n',theta_e(i),phi_e(i));
end


%% Evaluation
fprintf('\nResults evaluation:\n');
% Copy the real sources position
srcPos = zeros(3,2);
for i = 1:nsrcMixture
    srcPos(:,i) = sourcesStruct(i).source_xyz(:,1); % source_xyz(:,1) by default because the source is not moving
end

% Compute the microphone array centroid
micPosCentroid = mean(sensorsStruct.sensor_xyz,1)';

% Express srcPos in the microphone array referential
srcPos = bsxfun(@minus,srcPos,micPosCentroid);

% Compute azimuth and evelevation for reference sources (ground thruth)
[thetaRef,phiRef] = MBSS_true_aoa(srcPos);

% Compute and display metrics between ground thruth and estimated angles (Recall, Precision, F-measure, Accuracy)
for i = 1:nsrcLocalization
    [R, P, F, Acc, idCorrectEst] = MBSS_eval_angle([theta_e;phi_e], [thetaRef(i);phiRef(i)],angleThreshold);
    fprintf(' - Source %d:\n',i);
    if(R==1)
        fprintf('   - Recall: %d       (with estimated source number %d)\n',R,idCorrectEst);
    else
        fprintf('   - Recall: %d\n',R);
    end
    fprintf('   - Precision: %.2f (%d estimated sources)\n',P,nsrcLocalization);
    fprintf('   - F-measure: %.2f\n',F);
    fprintf('   - Accuracy:\n');
    fprintf('     - Azimuth (Theta): %.2f°\n     - Elevation (Phi): %.2f°\n     - Curvilinear:     %.2f° \n\n',Acc(1),Acc(2),Acc(3)');
end
