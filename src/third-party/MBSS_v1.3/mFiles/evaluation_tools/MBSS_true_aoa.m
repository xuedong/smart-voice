function [thetaRef,phiRef] = MBSS_true_aoa(srcPos)

% Function MBSS_true_aoa:
%
% Retrieve the true Angle Of Arrival corresponding to a given set 
% of source positions
%
% [thetaRef,phiRef] = MBSS_true_aoa(srcPos)
%
% Input:
% srcPos: 3 x nsrc matrix of cartesiant source positions expressed in the 
% microphone array referential (microphone array centroid).
%
% Output:
% thetaRef: 1 x nsrc vector of true source's azimuth (in degrees)
% phiRef: 1 x nsrc vector of the true source's elevation (in degrees)
%
% Version: v1.0
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2015 Ewen Camberlein and Romain Lebarbenchon
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
% If you find it useful, please cite the following reference:
% http://bass-db.gforge.inria.fr/bss_locate/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[thetaRef,phiRef,~] = cart2sph(srcPos(1,:),srcPos(2,:),srcPos(3,:));

% Back to deg
thetaRef = thetaRef.*180/pi; 
phiRef = phiRef.*180/pi;

return;