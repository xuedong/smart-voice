function specInst = MBSS_computeAngularSpectrum(x,fs,angularSpectrumMeth,c,micPos,thetaGrid,phiGrid,alphaRes)

% Function MBSS_computeAngularSpectrum
%
% Compute the time-frequency transform and the local angular spectrum
% method on the result.
%
% INPUT:
% x         : nsampl x nchan, matrix containing nchan time-domain mixture 
%             signals with nsampl samples
% fs        : 1 x 1, sampling frequency in Hz
% angularSpectrumMeth: string, angular spectrum method
% c         : 1 x 1, speed of sound (m/s)
% micPos    : nchan x 3, cartesian coordinates of the nchan microphones
% thetaGrid : 1 x nDirection, Azimuth grid
% phiGrid   : 1 x nDirection, Elevation grid
% alphaRes  : 1 x 1, interpolation resolution
%
% OUTPUT:
% specInst : 1 x nDirection, angular spectrum
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

wlen = 1024;
f = fs/wlen*(1:wlen/2).'; % frequency axis

if strfind(angularSpectrumMeth, 'GCC'),
    % Linear transform  
    X = MBSS_stft_multi(x.',wlen);
    X = X(2:end,:,:);
else
    % Quadratic transform
    hatRxx = MBSS_qstft_multi(x,fs,wlen,8,2);
    hatRxx = permute(hatRxx(:,:,2:end,:),[3 4 1 2]);
end


%% Computing the angular spectrum

switch angularSpectrumMeth
    case 'GCC-PHAT'
        specInst = GCC_PHAT_MULTI(X, f, c, micPos, thetaGrid, phiGrid, alphaRes);
    case 'GCC-NONLIN'
        specInst = GCC_NONLIN_MULTI(X, f, fs, c, micPos, thetaGrid, phiGrid, alphaRes);
    case 'MVDR'
        specInst = MVDR_MULTI(hatRxx, f, c, micPos, thetaGrid, phiGrid, alphaRes);
    case 'MVDRW'
        specInst = MVDRW_MULTI(hatRxx, f, c, micPos, thetaGrid, phiGrid, alphaRes);
    case 'DS'
        specInst = DS_MULTI(hatRxx, f, c, micPos, thetaGrid, phiGrid, alphaRes);
    case 'DSW'
        specInst = DSW_MULTI(hatRxx, f, c, micPos, thetaGrid, phiGrid, alphaRes);
    case 'DNM'
        specInst = DNM_MULTI(hatRxx, f, c, micPos, thetaGrid, phiGrid, alphaRes);
    case 'MUSIC'
        specInst = MUSIC_MULTI(hatRxx, f, c, micPos, thetaGrid, phiGrid, alphaRes);
end
end

%% MULTICHANNEL ANGULAR SPECTRUM METHODS

function [specInst] = GCC_PHAT_MULTI(X, f, c, micPos, thetaGrid, phiGrid, alphaRes)

% Function GCC_PHAT_MULTI
%
% Compute the GCC-PHAT algorithm for all pairs of microphones.
%
% INPUT:
% X : nfreq x nfram x N , N multichannel time-frequency transformed
% signals
% f : 1 x nfreq, frequency axis
% c : 1 x 1, speed of sound (in m/s)
% micPos : N x 3 , cartesian coordinates of the N microphones
% thetaGrid : 1 x nDirection, Azimuth grid
% phiGrid : 1 x nDirection, Elevation grid
% alphaRes : 1 x 1, interpolation resolution
%
% OUTPUT:
% specInst : 1 x nDirection, angular spectrum
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

%% pair search and grid adaptation to pairs referential
[pairId, ~, alpha, alphaSampled, tauGrid] = MBSS_preprocess(c, micPos, thetaGrid, phiGrid, alphaRes);

%% Computing the angular spectrum
[~,nFrames,~] = size(X);
nGrid = length(thetaGrid); % number of points of the "global" "sphere" grid
nPairs = size(pairId,1);
specEntireGrid = zeros(nGrid, nFrames, nPairs);

for i = 1:nPairs
    spec = phat_spec(X(:,:,pairId(i,:)), f, tauGrid{i}); % NV % [freq x fram x local angle for each pair]
    % sum on frequencies
    specSampledgrid = (shiftdim(sum(spec,1)))';
    % Order 1 interpolation on the entire grid
    specEntireGrid(:,:,i) = interp1q(alphaSampled{i}', specSampledgrid, alpha(i,:)'); % original
    
end

%% SRP: add the spec contribution of each microphone pair
specInst = sum(specEntireGrid,3);

end

function [specInst] = GCC_NONLIN_MULTI(X, f,fs, c, micPos, thetaGrid, phiGrid, alphaRes)

% Function GCC_NONLIN_MULTI
%
% Compute the GCC-NONLIN algorithm for all pairs of microphones.
%
% INPUT:
% X         : nfreq x nfram x N , N multichannel time-frequency transformed
%             signals
% f         : 1 x nfreq, frequency axis
% fs        : 1 x 1, sampling frequency (in Hz)
% c         : 1 x 1, speed of sound (in m/s)
% micPos    : N x 3 , cartesian coordinates of the N microphones
% thetaGrid : 1 x nDirection, Azimuth grid
% phiGrid   : 1 x nDirection, Elevation grid
% alphaRes  : 1 x 1, interpolation resolution
%
% OUTPUT:
% specInst : 1 x nDirection, angular spectrum
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

%% pair search and grid adaptation to pairs referential
[pairId, d, alpha, alphaSampled, tauGrid] = MBSS_preprocess(c, micPos, thetaGrid, phiGrid, alphaRes);

%% Computing the angular spectrum
% local spectrum
alpha_meth = 10*c./(d*fs);
[~,nFrames,~] = size(X);

nGrid = length(thetaGrid); % number of points of the "global" "sphere" grid
nPairs = size(pairId,1);
specEntireGrid = zeros(nGrid, nFrames, nPairs);

for i = 1:nPairs
    spec = nonlin_spec(X(:,:,pairId(i,:)), f, alpha_meth(i), tauGrid{i});
    % sum on frequencies
    specSampledgrid = (shiftdim(sum(spec,1)))';
    % Order 1 interpolation on the entire grid
    
    specEntireGrid(:,:,i) = interp1q(alphaSampled{i}', specSampledgrid, alpha(i,:)');
end

%% SRP: add the spec contribution of each microphone pair
specInst = sum(specEntireGrid,3);

end

function [specInst] = MVDR_MULTI(hatRxx, f, c, micPos, thetaGrid, phiGrid, alphaRes)

% Function MVDR_MULTI
%
% Compute the MVDR_SPEC algorithm for all pairs of microphones.
%
% INPUT:
% hatRxx    : nfreq x nfram x N x N , spatial covariance matrices in all 
%             time-frequency bins
% f         : 1 x nfreq, frequency axis
% c         : 1 x 1, speed of sound (in m/s)
% micPos    : N x 3 , cartesian coordinates of the N microphones
% thetaGrid : 1 x nDirection, Azimuth grid
% phiGrid   : 1 x nDirection, Elevation grid
% alphaRes  : 1 x 1, interpolation resolution
%
% OUTPUT:
% specInst : 1 x nDirection, angular spectrum
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

%% pair search and grid adaptation to pairs referential
[pairId, ~, alpha, alphaSampled, tauGrid] = MBSS_preprocess(c, micPos, thetaGrid, phiGrid, alphaRes);

%% Computing the angular spectrum
[~,nFrames,~,~] = size(hatRxx); % nbin x nFrames x 2 x 2
nGrid = length(thetaGrid); % number of points of the "global" "sphere" grid
nPairs = size(pairId,1);
specEntireGrid = zeros(nGrid, nFrames, nPairs);

for i = 1:nPairs
    spec = mvdr_spec(hatRxx(:,:,pairId(i,:),pairId(i,:)) , f, tauGrid{i}); %
    % sum on frequencies
    specSampledgrid = (shiftdim(sum(spec,1)))';
    % Order 1 interpolation on the entire grid
    specEntireGrid(:,:,i) = interp1q(alphaSampled{i}', specSampledgrid, alpha(i,:)');
end

%% add the spec contribution of each microphone pair
specInst = sum(specEntireGrid,3);
end

function [specInst] = MVDRW_MULTI(hatRxx, f, c, micPos, thetaGrid, phiGrid, alphaRes)

% Function MVDRW_MULTI
%
% Compute the MVDRW_SPEC algorithm for all pairs of microphones.
%
% INPUT:
% hatRxx    : nfreq x nfram x N x N , spatial covariance matrices in all 
%             time-frequency bins
% f         : 1 x nfreq, frequency axis
% c         : 1 x 1, speed of sound (in m/s)
% micPos    : N x 3 , cartesian coordinates of the N microphones
% thetaGrid : 1 x nDirection, Azimuth grid
% phiGrid   : 1 x nDirection, Elevation grid
% alphaRes  : 1 x 1, interpolation resolution
%
% OUTPUT:
% specInst : 1 x nDirection, angular spectrum
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

%% pair search and grid adaptation to pairs referential
[pairId, d, alpha, alphaSampled, tauGrid] = MBSS_preprocess(c, micPos, thetaGrid, phiGrid, alphaRes);

%% Computing the angular spectrum
% local spectrum
[~,nFrames,~,~] = size(hatRxx); % nbin x nFrames x 2 x 2
nGrid = length(thetaGrid); % number of points of the "global" "sphere" grid
nPairs = size(pairId,1);
specEntireGrid = zeros(nGrid, nFrames, nPairs);

for i = 1:nPairs
    spec = mvdrw_spec(hatRxx(:,:,pairId(i,:),pairId(i,:)) , f, d(i), tauGrid{i}); %
    % sum on frequencies
    specSampledgrid = (shiftdim(sum(spec,1)))';
    % Order 1 interpolation on the entire grid
    specEntireGrid(:,:,i) = interp1q(alphaSampled{i}', specSampledgrid, alpha(i,:)');
end

%% add the spec contribution of each microphone pair
specInst = sum(specEntireGrid,3);
end

function [specInst] = DS_MULTI(hatRxx, f, c, micPos, thetaGrid, phiGrid, alphaRes)

% Function DS_MULTI
%
% Compute the DS_SPEC algorithm for all pairs of microphones.
%
% INPUT:
% hatRxx    : nfreq x nfram x N x N , spatial covariance matrices in all 
%             time-frequency bins
% f         : 1 x nfreq, frequency axis
% c         : 1 x 1, speed of sound (in m/s)
% micPos    : N x 3 , cartesian coordinates of the N microphones
% thetaGrid : 1 x nDirection, Azimuth grid
% phiGrid   : 1 x nDirection, Elevation grid
% alphaRes  : 1 x 1, interpolation resolution
%
% OUTPUT:
% specInst : 1 x nDirection, angular spectrum
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

%% pair search and grid adaptation to pairs referential
[pairId, ~, alpha, alphaSampled, tauGrid] = MBSS_preprocess(c, micPos, thetaGrid, phiGrid, alphaRes);

%% Computing the angular spectrum
% local spectrum
[~,nFrames,~,~] = size(hatRxx); % nbin x nFrames x 2 x 2
nGrid = length(thetaGrid); % number of points of the "global" "sphere" grid
nPairs = size(pairId,1);
specEntireGrid = zeros(nGrid, nFrames, nPairs);

for i = 1:nPairs
    spec = ds_spec(hatRxx(:,:,pairId(i,:),pairId(i,:)) , f, tauGrid{i}); %
    % sum on frequencies
    specSampledgrid = (shiftdim(sum(spec,1)))';
    % Order 1 interpolation on the entire grid
    specEntireGrid(:,:,i) = interp1q(alphaSampled{i}', specSampledgrid, alpha(i,:)');
end

%% add the spec contribution of each microphone pair
specInst = sum(specEntireGrid,3);
end

function [specInst] = DSW_MULTI(hatRxx, f, c, micPos, thetaGrid, phiGrid, alphaRes)

% Function DSW_MULTI
%
% Compute the DSW_SPEC algorithm for all pairs of microphones.
%
% INPUT:
% hatRxx    : nfreq x nfram x N x N , spatial covariance matrices in all 
%             time-frequency bins
% f         : 1 x nfreq, frequency axis
% c         : 1 x 1, speed of sound (in m/s)
% micPos    : N x 3 , cartesian coordinates of the N microphones
% thetaGrid : 1 x nDirection, Azimuth grid
% phiGrid   : 1 x nDirection, Elevation grid
% alphaRes  : 1 x 1, interpolation resolution
%
% Outputs :
% specInst : 1 x nDirection, angular spectrum
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

%% pair search and grid adaptation to pairs referential
[pairId, d, alpha, alphaSampled, tauGrid] = MBSS_preprocess(c, micPos, thetaGrid, phiGrid, alphaRes);

%% Computing the angular spectrum
% local spectrum
[~,nFrames,~,~] = size(hatRxx); % nbin x nFrames x 2 x 2
nGrid = length(thetaGrid); % number of points of the "global" "sphere" grid
nPairs = size(pairId,1);
specEntireGrid = zeros(nGrid, nFrames, nPairs);

for i = 1:nPairs
    spec = dsw_spec(hatRxx(:,:,pairId(i,:),pairId(i,:)) , f, d(i), tauGrid{i}); %
    % sum on frequencies
    specSampledgrid = (shiftdim(sum(spec,1)))';
    % Order 1 interpolation on the entire grid
    specEntireGrid(:,:,i) = interp1q(alphaSampled{i}', specSampledgrid, alpha(i,:)');
end

%% add the spec contribution of each microphone pair
specInst = sum(specEntireGrid,3);
end

function [specInst] = DNM_MULTI(hatRxx, f, c, micPos, thetaGrid, phiGrid, alphaRes)

% Function DNM_MULTI
%
% Compute the DNM_SPEC algorithm for all pairs of microphones.
%
% INPUT:
% hatRxx    : nfreq x nfram x N x N , spatial covariance matrices in all 
%             time-frequency bins
% f         : 1 x nfreq, frequency axis
% c         : 1 x 1, speed of sound (in m/s)
% micPos    : N x 3 , cartesian coordinates of the N microphones
% thetaGrid : 1 x nDirection, Azimuth grid
% phiGrid   : 1 x nDirection, Elevation grid
% alphaRes  : 1 x 1, interpolation resolution
%
% OUTPUT:
% specInst : 1 x nDirection, angular spectrum
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

%% pair search and grid adaptation to pairs referential
[pairId, d, alpha, alphaSampled, tauGrid] = MBSS_preprocess(c, micPos, thetaGrid, phiGrid, alphaRes);

%% Computing the angular spectrum
% local spectrum
[~,nFrames,~,~] = size(hatRxx); % nbin x nFrames x 2 x 2
nGrid = length(thetaGrid); % number of points of the "global" "sphere" grid
nPairs = size(pairId,1);
specEntireGrid = zeros(nGrid, nFrames, nPairs);

for i = 1:nPairs
    spec = dnm_spec(hatRxx(:,:,pairId(i,:),pairId(i,:)) , f, d(i), tauGrid{i}); %
    % sum on frequencies
    specSampledgrid = (shiftdim(sum(spec,1)))';
    % Order 1 interpolation on the entire grid
    specEntireGrid(:,:,i) = interp1q(alphaSampled{i}', specSampledgrid, alpha(i,:)');
end

%% add the spec contribution of each microphone pair
specInst = sum(specEntireGrid,3);
end

function [specInst] = MUSIC_MULTI(hatRxx, f, c, micPos, thetaGrid, phiGrid, alphaRes)

% Function MUSIC_MULTI
%
% Compute the MUSIC_SPEC algorithm for all pairs of microphones.
%
% INPUT:
% hatRxx    : nfreq x nfram x N x N , spatial covariance matrices in all 
%             time-frequency bins
% f         : 1 x nfreq, frequency axis
% c         : 1 x 1, speed of sound (in m/s)
% micPos    : N x 3 , cartesian coordinates of the N microphones
% thetaGrid : 1 x nDirection, Azimuth grid
% phiGrid   : 1 x nDirection, Elevation grid
% alphaRes  : 1 x 1, interpolation resolution
%
% OUTPUT:
% specInst : 1 x nDirection, angular spectrum
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

%% pair search and grid adaptation to pairs referential
[pairId, ~, alpha, alphaSampled, tauGrid] = MBSS_preprocess(c, micPos, thetaGrid, phiGrid, alphaRes);

%% Computing the angular spectrum
% local spectrum
[~,nFrames,~,~] = size(hatRxx); % nbin x nFrames x 2 x 2
nGrid = length(thetaGrid); % number of points of the "global" "sphere" grid
nPairs = size(pairId,1);
specEntireGrid = zeros(nGrid, nFrames, nPairs);

for i = 1:nPairs
    spec = music_spec(hatRxx(:,:,pairId(i,:),pairId(i,:)) , f, tauGrid{i}); %
    % sum on frequencies
    specSampledgrid = (shiftdim(sum(spec,1)))';
    % Order 1 interpolation on the entire grid
    specEntireGrid(:,:,i) = interp1q(alphaSampled{i}', specSampledgrid, alpha(i,:)');
end

%% add the spec contribution of each microphone pair
specInst = sum(specEntireGrid,3);
end

%% BSS-LOCATE ANGULAR SPECTRUM FUNCTIONS (2 CHANNELS)

function spec = phat_spec(X, f, tauGrid)

% PHAT_SPEC Computes the GCC-PHAT spectrum as defined in
% C. Knapp, G. Carter, "The generalized cross-correlation method for
% estimation of time delay", IEEE Transactions on Acoustics, Speech and
% Signal Processing, 24(4):320â€“327, 1976.
%
% spec = phat_spec(X, f, tauGrid)
%
% Inputs:
% X: nbin x nFrames x 2 matrix containing the STFT coefficients of the input
%     signal in all time-frequency bins
% f: nbin x 1 vector containing the center frequency of each frequency bin
%     in Hz
% tauGrid: 1 x ngrid vector of possible TDOAs in seconds
%
% Output:
% spec: nbin x nFrames x ngrid array of angular spectrum values
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2010-2011 Charles Blandin and Emmanuel Vincent
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
% If you find it useful, please cite the following reference:
% Charles Blandin, Emmanuel Vincent and Alexey Ozerov, "Multi-source TDOA
% estimation in reverberant audio using angular spectra and clustering",
% Signal Processing 92, pp. 1950-1960, 2012.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X1 = X(:,:,1);
X2 = X(:,:,2);

[nbin,nFrames] = size(X1);
ngrid = length(tauGrid);

spec = zeros(nbin,nFrames,ngrid);
P = X1.*conj(X2);
P = P./abs(P);

temp = ones(1,nFrames);
for pkInd = 1:ngrid,
    EXP = exp(-2*1i*pi*tauGrid(pkInd)*f);
    EXP = EXP(:,temp);
    spec(:,:,pkInd) = real(P.*EXP);
end

end

function spec = nonlin_spec(X, f, alpha, tauGrid)

% NONLIN_SPEC Computes the nonlinear GCC-PHAT spectrum defined in
% B. Loesch, B. Yang, "Blind source separation based on time-frequency
% sparseness in the presence of spatial aliasing", in 9th Int. Conf. on
% Latent Variable Analysis and Signal Separation (LVA/ICA), pp. 1â€“8, 2010.
%
% spec = nonlin_spec(X, f, alpha, tauGrid)
%
% Inputs:
% X: nbin x nFrames x 2 matrix containing the STFT coefficients of the input
%     signal in all time-frequency bins
% f: nbin x 1 vector containing the center frequency of each frequency bin
%     in Hz
% alpha: nonlinearity parameter
% tauGrid: 1 x ngrid vector of possible TDOAs in seconds
%
% Output:
% spec: nbin x nFrames x ngrid array of angular spectrum values
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2010-2011 Charles Blandin and Emmanuel Vincent
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
% If you find it useful, please cite the following reference:
% Charles Blandin, Emmanuel Vincent and Alexey Ozerov, "Multi-source TDOA
% estimation in reverberant audio using angular spectra and clustering",
% Signal Processing 92, pp. 1950-1960, 2012.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X1 = X(:,:,1);
X2 = X(:,:,2);

[nbin,nFrames] = size(X1);
ngrid = length(tauGrid);

spec = zeros(nbin,nFrames,ngrid);
P = X1.*conj(X2);
P = P./abs(P);

temp = ones(1,nFrames);
for pkInd = 1:ngrid,
    EXP = exp(-2*1i*pi*tauGrid(pkInd)*f);
    EXP = EXP(:,temp);
    spec(:,:,pkInd) = 1 - tanh(alpha*sqrt(abs(2-2*real(P.*EXP)))); % RLB : la valeur absolu permet de ne pas se retrouver avec des spec complexe (déjà observé). Je ne connais pas les tenants et aboutissants de ce pb.
end

end

function spec = mvdr_spec(hatRxx, f, tauGrid)


% MVDR_SPEC Computes the SNR in all directions using the MVDR beamformer
%
% spec = mvdr_spec(hatRxx, f, tauGrid)
%
% Inputs:
% hatRxx : nbin x nFrames x 2 x 2 array containing the spatial covariance
%     matrices of the input signal in all time-frequency bins
% f: nbin x 1 vector containing the center frequency of each frequency bin
%     in Hz
% tauGrid: 1 x ngrid vector of possible TDOAs in seconds
%
% Output:
% spec: nbin x nFrames x ngrid array of SNR values
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2010-2011 Charles Blandin and Emmanuel Vincent
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
% If you find it useful, please cite the following reference:
% Charles Blandin, Emmanuel Vincent and Alexey Ozerov, "Multi-source TDOA
% estimation in reverberant audio using angular spectra and clustering",
% Signal Processing 92, pp. 1950-1960, 2012.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[nbin,nFrames] = size(hatRxx(:,:,1,1));
ngrid = length(tauGrid);
R11 = hatRxx(:,:,1,1);
R12 = hatRxx(:,:,1,2);
R21 = hatRxx(:,:,2,1);
R22 = hatRxx(:,:,2,2);
TR = real(R11 + R22);

SNR = zeros(nbin,nFrames,ngrid);
for pkInd=1:ngrid,
    EXP = repmat(exp(-2*1i*pi*tauGrid(pkInd)*f),1,nFrames);
    NUM = real(R11.*R22 - R12.*R21)./(TR - 2*real(R12.*EXP));
    SNR(:,:,pkInd) = NUM./(.5*TR-NUM);
end
spec = SNR;

end

function spec = mvdrw_spec(hatRxx, f, d, tauGrid)

% MVDRW_SPEC Computes the SNR in all directions using the MVDR beamformer
% and frequency weighting
%
% spec = mvdrw_spec(hatRxx, f, d, tauGrid)
%
% Inputs:
% hatRxx : nbin x nFrames x 2 x 2 array containing the spatial covariance
%     matrices of the input signal in all time-frequency bins
% f: nbin x 1 vector containing the center frequency of each frequency bin
%     in Hz
% d: microphone spacing in meters
% tauGrid: 1 x ngrid vector of possible TDOAs in seconds
%
% Output:
% spec: nbin x nFrames x ngrid array of SNR values
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2010-2011 Charles Blandin and Emmanuel Vincent
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
% If you find it useful, please cite the following reference:
% Charles Blandin, Emmanuel Vincent and Alexey Ozerov, "Multi-source TDOA
% estimation in reverberant audio using angular spectra and clustering",
% Signal Processing 92, pp. 1950-1960, 2012.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[nbin,nFrames] = size(hatRxx(:,:,1,1));
ngrid = length(tauGrid);
R11 = hatRxx(:,:,1,1);
R12 = hatRxx(:,:,1,2);
R21 = hatRxx(:,:,2,1);
R22 = hatRxx(:,:,2,2);
TR = real(R11 + R22);
c = 343;
SINC = sinc(2*f*d/c);

SNR = zeros(nbin,nFrames,ngrid);
for pkInd=1:length(tauGrid),
    EXP = repmat(exp(-2*1i*pi*tauGrid(pkInd)*f),1,nFrames);
    NUM = real(R11.*R22 - R12.*R21)./(TR - 2*real(R12.*EXP));
    SNR(:,:,pkInd) = repmat(-(1+SINC)/2,1,nFrames) + repmat((1-SINC)/2,1,nFrames).*NUM./(.5*TR-NUM);
end
spec = SNR;

end

function spec = ds_spec(hatRxx, f, tauGrid)

% DS_SPEC Computes the SNR in all directions using the DS beamformer
%
% spec = ds_spec(hatRxx, f, tauGrid)
%
% Inputs:
% hatRxx : nbin x nFrames x 2 x 2 array containing the spatial covariance
%     matrices of the input signal in all time-frequency bins
% f: nbin x 1 vector containing the center frequency of each frequency bin
%     in Hz
% tauGrid: 1 x ngrid vector of possible TDOAs in seconds
%
% Output:
% spec: nbin x nFrames x ngrid array of SNR values
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2010-2011 Charles Blandin and Emmanuel Vincent
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
% If you find it useful, please cite the following reference:
% Charles Blandin, Emmanuel Vincent and Alexey Ozerov, "Multi-source TDOA
% estimation in reverberant audio using angular spectra and clustering",
% Signal Processing 92, pp. 1950-1960, 2012.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[nbin,nFrames] = size(hatRxx(:,:,1,1));
ngrid = length(tauGrid);
R11 = hatRxx(:,:,1,1);
R12 = hatRxx(:,:,1,2);
R22 = hatRxx(:,:,2,2);
TR = real(R11 + R22);

SNR = zeros(nbin,nFrames,ngrid);
for pkInd=1:ngrid,
    EXP = repmat(exp(-2*1i*pi*tauGrid(pkInd)*f),1,nFrames);
    SNR(:,:,pkInd) = (TR + 2*real(R12.*EXP))./(TR - 2*real(R12.*EXP));
end
spec = SNR;

end

function spec = dsw_spec(hatRxx, f, d, tauGrid)

% DSW_SPEC Computes the SNR in all directions using the DS beamformer and
% frequency weighting
%
% spec = dsw_spec(hatRxx, f, d, tauGrid)
%
% Inputs:
% hatRxx : nbin x nFrames x 2 x 2 array containing the spatial covariance
%     matrices of the input signal in all time-frequency bins
% f: nbin x 1 vector containing the center frequency of each frequency bin
%     in Hz
% d: microphone spacing in meters
% tauGrid: 1 x ngrid vector of possible TDOAs in seconds
%
% Output:
% spec: nbin x nFrames x ngrid array of SNR values
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2010-2011 Charles Blandin and Emmanuel Vincent
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
% If you find it useful, please cite the following reference:
% Charles Blandin, Emmanuel Vincent and Alexey Ozerov, "Multi-source TDOA
% estimation in reverberant audio using angular spectra and clustering",
% Signal Processing 92, pp. 1950-1960, 2012.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[nbin,nFrames] = size(hatRxx(:,:,1,1));
ngrid = length(tauGrid);
R11 = hatRxx(:,:,1,1);
R12 = hatRxx(:,:,1,2);
R22 = hatRxx(:,:,2,2);
TR = real(R11 + R22);
c = 343;
SINC = sinc(2*f*d/c);

SNR = zeros(nbin,nFrames,ngrid);
for pkInd=1:ngrid,
    EXP = repmat(exp(-2*1i*pi*tauGrid(pkInd)*f),1,nFrames);
    SNR(:,:,pkInd) = repmat(-(1+SINC)/2,1,nFrames) + repmat((1-SINC)/2,1,nFrames).*(TR + 2*real(R12.*EXP))./(TR - 2*real(R12.*EXP));
end
spec = SNR;

end

function spec = dnm_spec(hatRxx, f, d, tauGrid)

% APR_SPEC Computes the SNR in all directions using ML under a diffuse
% noise model
%
% spec = dnm_spec(hatRxx, f, d, tauGrid)
%
% Inputs:
% hatRxx : nbin x nFrames x 2 x 2 array containing the spatial covariance
%     matrices of the input signal in all time-frequency bins
% f: nbin x 1 vector containing the center frequency of each frequency bin
%     in Hz
% d: microphone spacing in meters
% tauGrid: 1 x ngrid vector of possible TDOAs in seconds
%
% Output:
% spec: nbin x nFrames x ngrid array of SNR values
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2010-2011 Charles Blandin and Emmanuel Vincent
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
% If you find it useful, please cite the following reference:
% Charles Blandin, Emmanuel Vincent and Alexey Ozerov, "Multi-source TDOA
% estimation in reverberant audio using angular spectra and clustering",
% Signal Processing 92, pp. 1950-1960, 2012.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[nbin,nFrames] = size(hatRxx(:,:,1,1));
ngrid = length(tauGrid);
R11 = real(hatRxx(:,:,1,1));
R12 = hatRxx(:,:,1,2);
R21 = hatRxx(:,:,2,1);
R22 = real(hatRxx(:,:,2,2));
c = 343;
SINC = sinc(2*f*d/c);
SINC2 = SINC.^2;

% Initializing the variances
vs = zeros(nbin,nFrames,ngrid);
vb = zeros(nbin,nFrames,ngrid);
for pkInd = 1:ngrid,
    
    % Computing inv(A) = [invA11 invA12; conj(invA11) -invA12]
    EXP = exp(-2*1i*pi*tauGrid(pkInd)*f);
    P = SINC .* EXP;
    invA11 = sqrt(.5)./(1-real(P)).*(1-conj(P));
    invA12 = -(1-P)./(SINC-EXP).*invA11;
    
    % Computing inv(Lambda) = [.5 invL12; 0 invL22]
    DEN = .5./(1-2*real(P)+SINC2);
    invL12 = (SINC2-1).*DEN;
    invL22 = 2*(1-real(P)).*DEN;
    
    % Computing vs and vb without nonnegativity constraint
    ARA1 = repmat(abs(invA11).^2,1,nFrames).*R11 + repmat(abs(invA12).^2,1,nFrames).*R22;
    ARA2 = ARA1 - 2 * real(repmat(invA11.*invA12,1,nFrames).*R21);
    ARA1 = ARA1 + 2 * real(repmat(invA11.*conj(invA12),1,nFrames).*R12);
    vsind = .5*ARA1 + repmat(invL12,1,nFrames).*ARA2;
    vbind = repmat(invL22,1,nFrames).*ARA2;
    
    % Enforcing the nonnegativity constraint (on vs only)
    neg = (vsind < 0) | (vbind < 0);
    vsind(neg) = 0;
    vs(:,:,pkInd) = vsind;
    vb(:,:,pkInd) = vbind;
end
spec = vs./vb;

end

function spec = music_spec(hatRxx, f, tauGrid)

% MUSIC_SPEC Computes the MUSIC spectrum as defined in
% R. Schmidt, "Multiple emitter location and signal parameter estimation",
% IEEE Transactions on Antennas and Propagation, 34(3):276â€“280, 1986.
%
% spec = music_spec(hatRxx, f, tauGrid)
%
% hatRxx : nbin x nFrames x 2 x 2 array containing the spatial covariance
%     matrices of the input signal in all time-frequency bins
% f: nbin x 1 vector containing the center frequency of each frequency bin
%     in Hz
% tauGrid: 1 x ngrid vector of possible TDOAs in seconds
%
% Output:
% spec: nbin x nFrames x ngrid array of SNR values
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2010-2011 Charles Blandin and Emmanuel Vincent
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
% If you find it useful, please cite the following reference:
% Charles Blandin, Emmanuel Vincent and Alexey Ozerov, "Multi-source TDOA
% estimation in reverberant audio using angular spectra and clustering",
% Signal Processing 92, pp. 1950-1960, 2012.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[nbin,nFrames] = size(hatRxx(:,:,1,1));
ngrid = length(tauGrid);
R11 = real(hatRxx(:,:,1,1));
R12 = hatRxx(:,:,1,2);
R22 = real(hatRxx(:,:,2,2));
TR = R11 + R22;
DET = R11.*R22 - abs(R12).^2;
lambda = .5*(TR + sqrt(TR.^2 - 4*DET));
V2 = (lambda-R11)./R12;

spec = zeros(nbin,nFrames,ngrid);
for pkInd = 1:ngrid,
    EXP = repmat(exp(-2*1i*pi*tauGrid(pkInd)*f),1,nFrames);
    spec(:,:,pkInd) = 1 ./ (1 - .5 * abs(1 + V2 .* conj(EXP)).^2./(1 + abs(V2).^2));
end

end