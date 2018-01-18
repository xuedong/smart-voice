function [theta_e, phi_e] = MBSS_locate_spec(x, fs, nsrc , micPos, c, angularSpectrumMeth, pooling, thetaBound, phiBound, thetaPhiRes, alphaRes, minAngle, normalizeSpecInst, specDisplay)

% Function MBSS_locate_spec 
% Estimate sources localization in a multichannel convolutive mixture using
% an angular spectrum based approach
%
% Mandatory Inputs:
% x      : nsampl x nchan, matrix containing nchan time-domain mixture 
%          signals with nsampl samples
% fs     : 1 x 1, sampling frequency in Hz
% micPos : nchan x 3, cartesian coordinates of the nchan microphones
% nsrc   : 1 x 1, maximal number of sources
%
% Optional Inputs:
% c                   : 1 x 1, speed of sound (m/s). Default : 343 m/s
% angularSpectrumMeth : string, local angular spectrum method used :
%                       'GCC-PHAT'(default) 'GCC-NONLIN'  'MVDR' 'MVDRW' 'DS' 'DSW' 'DNM' 'MUSIC'
% pooling             : string, pooling function: 'max' (default) or 'sum'
% thetaBound          : 1 x 2, azimuth boundaries (degrees). Default : [-179 180]
% phiBound            : 1 x 2, elevation boundaries (degrees). Default : [-90 90]
% thetaPhiRes         : 1 x 1, sampling resolution applied to azimuth and elevation parameters (degrees); Default value: 1°
% alphaRes            : 1 x 1, sampling resolution applied to each microphone pair referential (degrees); Default value: 5°
% minAngle            : 1 x 1, minimum distance (degrees) between two peaks. Default value: 1°;
% normalizeSpecInst   : 1 x 1 , flag used to activate normalization of instantaneous local angular spectra. Default: 0
% specDisplay         : 1 x 1 , flag used to display the global angular spectrum and directions of sources found. Default: 0.
%
% Outputs:
% theta_e : 1 x nSrcFound, vector of estimated azimuths (nSrcFound <= nsrc)
% phi_e   : 1 x nSrcFound, vector of estimated elevations (nSrcFound <= nsrc)
%
% Version : v1.2
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2015 Ewen Camberlein and Romain Lebarbenchon
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
% If you find it useful, please cite the following reference:
% http://bass-db.gforge.inria.fr/bss_locate/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Errors and default values
if nargin<4, error('Not enough input arguments.'); end
[nSamples,nChan]=size(x);
[nMic,~] = size(micPos);
fprintf('Input signal duration: %.02f seconds\n',nSamples/fs);
if nChan>nSamples, error('The input signal must be in columns.'); end
if nChan~=nMic, error('Number of microphones and number of signal channels must be the same'); end
if nargin < 5, c = 343; end
if nargin < 6, angularSpectrumMeth = 'GCC-PHAT'; end
if ~any(strcmp(angularSpectrumMeth, {'GCC-PHAT' 'GCC-NONLIN' 'MVDR' 'MVDRW' 'DS' 'DSW' 'DNM' 'MUSIC'})), error('Unknown local angular spectrum.'); end
if nargin < 7, pooling = 'max'; end
if ~any(strcmp(pooling, {'max' 'sum'})), error('Unknown pooling function.'); end
if nargin < 8, thetaBound = [-179 180]; end
if (length(thetaBound) ~= 2), error('Length of thetaBound must be 2'); end
if (thetaBound(1) >= thetaBound(2)), error('thetaBound must be filled in ascending order'); end
if nargin < 9, phiBound = [-90 90]; end
if (length(phiBound) ~= 2), error('Length of phiBound must be 2'); end
if (phiBound(1) >= phiBound(2)), error('phiBound must be filled in ascending order'); end
if nargin < 10, thetaPhiRes = 1; end
if nargin < 11, alphaRes = 5; end
if nargin < 12, minAngle = 1; end
if(minAngle < thetaPhiRes && nscr>1), error('Minimum angle between two peaks has to be upper than theta/phi resolution'); end
if nargin < 13, normalizeSpecInst = 0; end
if nargin < 14, specDisplay = 0; end

if((nsrc == 1) && normalizeSpecInst)
    warning('Use of instantaneous local angular spectra normalization with one source to be located is unnecessary. Switch to no normalization usage.');
    normalizeSpecInst = 0;
end


%% Generate search space : all combination of {theta,phi} resp. azimuth and elevation with two 1D vectors.
thetas = (thetaBound(1) : thetaPhiRes : thetaBound(2))';
phis   = (phiBound(1) : thetaPhiRes : phiBound(2));
nThetas = length(thetas);
nPhis = length(phis);
thetaGrid = repmat(thetas,nPhis,1)';
phiGrid   = (reshape(repmat(phis,nThetas,1),1,nThetas*nPhis));

%% Compute the angular spectrum
specInst = MBSS_computeAngularSpectrum(x, fs,angularSpectrumMeth, c, micPos, thetaGrid, phiGrid, alphaRes);

%% Normalize instantaneous local angular spectra if requested
if(normalizeSpecInst)
    [~,nFrames,~] = size(specInst);
    for i=1:nFrames
        minVal = min(min(specInst(:,i)));
        specInst(:,i)=(specInst(:,i) - minVal)/ max(max(specInst(:,i)- minVal));
    end
end

%% Pooling
switch pooling
    case 'max'
        specGlobal = shiftdim(max(specInst,[],2));
    case 'sum'
        specGlobal = shiftdim(sum(specInst,2));
end

%% Peak finding
[pfEstAngles] = MBSS_findPeaks2D(specGlobal, thetas, phis, thetaGrid, phiGrid, nsrc, minAngle,angularSpectrumMeth, specDisplay);
theta_e = pfEstAngles(:,1)';
phi_e = pfEstAngles(:,2)';


end

%% PEAKS FINDING METHODS
function [pfEstAngles] = MBSS_findPeaks2D(ppfSpec, piThetas, piPhis, piThetaGrid, piPhiGrid, iNbSources, iMinAngle,angularSpectrumMeth, bDisplayResults)

% Function MBSS_findPeaks2D
%
% This function search peaks in computed angular spectrum with respect to iNbSources and iMinAngle.
%
% INPUT:
% ppfSpec         : 1 x iNGridDirection : 1D angular spectrum
% piThetas        : 1 x iNbThetas : Azimuth sampled values
% piPhis          : 1 x iNbPhis : Elevation sampled values
% piThetaGrid     : 1 x iNGridDirection : Azimuth grid
% piPhiGrid       : 1 x iNGridDirection : Elevation grid
% iNbSources      : 1 x 1 : Number of sources to be found
% iMinAngle       : 1 x 1 : Minimum angle between two sources
% bDisplayResults : 1 x1 :display global 2D angular spectrum with sources found
%
% OUTPUT:
% pfEstAngles : iNbSourcesFound x 2 : azimuth and elevation for all sources found
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

iNbThetas = length(piThetas);
iNbPhis = length(piPhis);

% Convert angular spectrum in 2D
%fprintf('Size of ppfSpec:\n')
%disp(size(ppfSpec))
%fprintf('nbThetas: %d, nbphis: %d', iNbThetas, iNbPhis)
ppfSpec2D = (reshape(ppfSpec,iNbThetas,iNbPhis))';

% Estimate angular spectrum in theta and phi independently by taking
% the max in the corresponding direction
spec_theta_max = max(ppfSpec2D,[],1);
spec_phi_max   = max(ppfSpec2D,[],2);

if(iNbSources == 1)
    % Find the maximum peak both in theta and phi direction
    [~, iThetaId] = max(spec_theta_max);
    [~, iPhiId] = max(spec_phi_max);
    pfEstAngles = [piThetas(iThetaId) , piPhis(iPhiId)];
else
    % search all local maxima (local maximum : value higher than all neighborhood values)
    % some alternative implementations using matlab image processing toolbox are explained here :
    % http://stackoverflow.com/questions/22218037/how-to-find-local-maxima-in-image)
    
	% Current implementation uses no specific toolbox. Explanations can be found with following link : 
    % http://stackoverflow.com/questions/5042594/comparing-matrix-element-with-its-neighbours-without-using-loop-in-matlab
    % All values of flat peaks are detected as peaks with this implementation :
    ppfPadpeakFilter = ones(size(ppfSpec2D,1)+2,size(ppfSpec2D,2)+2) * -Inf;
    ppfPadpeakFilter(2:end-1,2:end-1) = ppfSpec2D;
    
    % Find peaks : compare values with their neighbours 
    ppiPeaks = ppfPadpeakFilter(2:end-1,2:end-1) >= ppfPadpeakFilter(1:end-2,2:end-1) & ... % top
               ppfPadpeakFilter(2:end-1,2:end-1) >= ppfPadpeakFilter(3:end,  2:end-1) & ... % bottom
               ppfPadpeakFilter(2:end-1,2:end-1) >= ppfPadpeakFilter(2:end-1,1:end-2) & ... % right 
               ppfPadpeakFilter(2:end-1,2:end-1) >= ppfPadpeakFilter(2:end-1,3:end)   & ... % left
               ppfPadpeakFilter(2:end-1,2:end-1) >= ppfPadpeakFilter(1:end-2,1:end-2) & ... % top/left
               ppfPadpeakFilter(2:end-1,2:end-1) >= ppfPadpeakFilter(1:end-2,3:end)   & ... % top/right
               ppfPadpeakFilter(2:end-1,2:end-1) >= ppfPadpeakFilter(3:end,  1:end-2) & ... % bottom/left 
               ppfPadpeakFilter(2:end-1,2:end-1) >= ppfPadpeakFilter(3:end,  3:end);        % bottom/right                 

    % number of local maxima
    iNbLocalmaxima = sum(sum(ppiPeaks));
    
    % local maxima with corrresponding values
    ppfSpec2D_peaks = (ppfSpec2D - min(min(ppfSpec2D))) .* ppiPeaks; % substract min value : avoid issues (when sorting peaks) if some peaks values are negatives
    
    % sort values of local maxima
    pfSpec1D_peaks= reshape(ppfSpec2D_peaks',1,iNbPhis*iNbThetas);
    [~,piIndexPeaks1D] = sort(pfSpec1D_peaks,'descend');
    
    piEstSourcesIndex = piIndexPeaks1D(1);  % first source is the global maximum (first one in piSortedPeaksIndex1D)
    index = 2; % search index in piSortedPeaksIndex1D
    iNbSourcesFound = 1; % set to one as global maximum is already selected as source
    
    %Filter the list of peaks found with respect to minAngle parameter
    while (iNbSourcesFound < iNbSources && index <= iNbLocalmaxima)
     
        bAngleAllowed = 1;
        % verify that current direction is allowed with respect to minAngle and sources already selected
        for i = 1:length(piEstSourcesIndex)
            
            % distance calculated using curvilinear abscissa (degrees) - ref. : http://geodesie.ign.fr/contenu/fichiers/Distance_longitude_latitude.pdf
            dist=acosd(sind(piPhiGrid(piEstSourcesIndex(i)))*sind(piPhiGrid(piIndexPeaks1D(index)))+cosd(piPhiGrid(piEstSourcesIndex(i)))*cosd(piPhiGrid(piIndexPeaks1D(index)))*cosd(piThetaGrid(piIndexPeaks1D(index))-piThetaGrid(piEstSourcesIndex(i))) );
            
            if(dist <iMinAngle)
                bAngleAllowed =0;
                break;
            end
        end
        
        % store new source
        if(bAngleAllowed)
            piEstSourcesIndex = [piEstSourcesIndex,piIndexPeaks1D(index)];
            iNbSourcesFound = iNbSourcesFound +1;            
        end
        
        index = index + 1;
    end
    
    pfEstAngles = [piThetaGrid(piEstSourcesIndex)' piPhiGrid(piEstSourcesIndex)'];
end

%% Display results
if (bDisplayResults)
    figHandle=figure;
    colormap(jet); % bleu jaune rouge
    imagesc(piThetas,piPhis,ppfSpec2D);
    set(gca,'YDir','normal');  
    hold on;    
    
    %display sources found
    for i =1:length(pfEstAngles(:,1))
        handle=plot(pfEstAngles(i,1),pfEstAngles(i,2),'*k','MarkerSize',15,'linewidth',1.5);    
    end   
    
    xlabel('\theta (degrees)');
    ylabel('\phi (degrees)');
    hold off;
    title(['\Phi^{' angularSpectrumMeth '}(\theta,\phi)   |  markers : sources found ']);
    
    print(figHandle,'resultsFigure.png','-dpng');
end

end

