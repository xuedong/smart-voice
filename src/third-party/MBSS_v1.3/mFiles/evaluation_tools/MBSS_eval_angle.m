function [R, P, F, Acc, idCorrectEst] = MBSS_eval_angle(loc_e, loc_true,thres)

% EVAL_ANGLE Evaluation of Angle estimation in terms of recall, precision,
% F-measure and accuracy
%
% [R, P, F, Acc, idCorrectEst] = MBSS_eval_angle(angle_e, angle_true, thresh)
%
% Inputs:
% loc_e: 2 x nsrce vector of estimated angles (azimuth and elevation)
% loc_true: 2 x 1 vector of true angle (azimuth and elevation)
% thresh: correctness threshold in degrees under the far-field assumption
%
% Outputs:
% R: recall
% P: precision
% F: F-measure
% Acc: 3 x 1 vector of estimated source accuracie in degrees  (+inf 
% if above the threshold) resp. in azimuth, elevation and curvilinear abscissa
% idCorrectEst : Position id of the first correct estimated source considering
% current true source (-1 if nos estimated sources retained)
%
% Version: v1.3
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2010-2011 Charles Blandin and Emmanuel Vincent
% Copyrigth 2015 Ewen Camberlein and Romain Lebarbenchon
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
% If you find it useful, please cite the following reference:
% - Charles Blandin, Emmanuel Vincent and Alexey Ozerov, "Multi-source TDOA
%   estimation in reverberant audio using angular spectra and clustering",
%   Signal Processing, to appear.
% - http://bass-db.gforge.inria.fr/bss_locate/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(size(loc_true,2) > 1 )
   error('Only evaluation of one real source is available');
end
if(size(loc_e,1) ~= 2 || size(loc_true,1) ~=2)
   error('Each localization must have two components (theta and phi)'); 
end

nsrce = size(loc_e,2);
nsrc = 1;

% angleError : [AzimuthError;ElevationError;Curvilinear abscissa error] for each estimated sources
% compute AzimuthError and ElevationError
angleErr = abs(bsxfun(@minus,loc_e,loc_true));
angleErr = bsxfun(@min,angleErr,360-angleErr);
% compute Curvilinear abscissa error
angleErr = [angleErr;acosd(sind(loc_e(2,:)).*sind(loc_true(2))+cosd(loc_e(2,:)).*cosd(loc_true(2)).*cosd(loc_true(1)-loc_e(1,:)))];


% correct position vs distance threshold is computed using curvilinear abscissa distance
correct = (angleErr(3,:) <= thres);
posCorrect = find(correct == 1);

% This code evaluates one real source.
% Therefore if one estimated source (among other estimated sources) respect
% the theshold constraint, we consider the estimation as correct and recall is set to "1"

if(sum(correct) >1)
    posCorrect = posCorrect(1); % The first index found among nsrce estimated correct sources corresponds to the highest peak
    correct = 1;
end

% Compute the evaluation metrics
%Recall
R = sum(correct)/nsrc;
%Precision
P = sum(correct)/nsrce;
% F-measure
F = 2*(P.*R)./(P + R + realmin);
% Accuracy

if(isempty(posCorrect))
    % No estimated source respects thres 
    idCorrectEst = -1;
    Acc = inf(3,1);
else
    Acc = angleErr(:,posCorrect);
    idCorrectEst = posCorrect;
end

return;
