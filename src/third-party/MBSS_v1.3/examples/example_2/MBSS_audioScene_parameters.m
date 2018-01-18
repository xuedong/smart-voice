function [roomStruct,sensorsStruct,sourcesStruct] = MBSS_audioScene_parameters()

% Function MBSS_audioScene_parameters
%
% This function store and display audio scene parameters in dedicated 
% structures for mixture generation
%
% OUTPUT:
% roomStruct : struct, room parameters used in the roomsimove toolbox
% sensorsStruct : struct, sensors parameters used in the roomsimove toolbox
% sourcesStruct : struct, sources parameters used in the roomsimove toolbox

% Version : v1.3
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2015 Ewen Camberlein and Romain Lebarbenchon
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
% If you find it useful, please cite the following reference:
% http://bass-db.gforge.inria.fr/bss_locate/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% display configuration
bDisplayConfiguration = 1;


%% Room configuration
% Structure fields declaration
roomStruct = struct(...
    'room_size',   NaN,...
    'F_abs',       NaN, ...
    'A',           NaN ...
    );

% Room size (in meters)
roomStruct.room_size = [6.046	4.562	2.535];

% Frequencies to define absorption coefficients
roomStruct.F_abs = [125	250	500	1000 2000 4000 8000];

% Absorption coefficients for surfaces x=0, x=Lx, y=0, y=Ly, z=0, z=Lz
roomStruct.A = ...
   [0.9 0.9	0.9	0.9	0.9	0.9	0.9;  %Ax1
	0.9	0.9	0.9	0.9	0.9	0.9	0.9;  %Ax2
	0.9	0.9	0.9	0.9	0.9	0.9	0.9;  %Ay1
	0.9	0.9	0.9	0.9	0.9	0.9	0.9;  %Ay2
	0.9	0.9	0.9	0.9	0.9	0.9	0.9;  %Az1
	0.9	0.9	0.9	0.9	0.9	0.9	0.9]; %Az2

%% Sensors Configuration
% Structure fields declaration
sensorsStruct = struct(...
    'sensor_nb',    NaN, ...
    'sensor_xyz',   NaN, ...
    'sensor_off',   NaN, ...
    'sensor_type',  NaN ...
    );

% Number of sensors
sensorsStruct.sensor_nb = 8;

% Sensor positions (in meters)
sensorsStruct.sensor_xyz = ...
  ...  x      y       z  
    [2.742	2.582	1.155;  %mic 1
	 2.671	2.582	1.231;  %mic 2
	 2.649	2.563	1.155;  %mic 3
	 2.649	2.492	1.231;  %mic 4
	 2.668	2.470	1.155;  %mic 5
	 2.739	2.470	1.231;  %mic 6
	 2.761	2.489	1.155;  %mic 7
	 2.761	2.560	1.231]; %mic 8

% Sensor directions (azimuth, elevation and roll offset in degrees, positive for slew left, nose up or right wing down)
sensorsStruct.sensor_off = zeros(sensorsStruct.sensor_nb,3); % no direction as all microphone as omnidirectionnal.

% Sensor direction-dependent impulse responses (e.g. gains or HRTFs)
sensorsStruct.sensor_type = [1 1 1 1 1 1 1 1]; % 1 = omnidirectional / 2 = cardioid

%% Sources configuration (1 structure per source) : example for two sources
% Comment : Roomsimove is able to handle moving source by sampling source positions
% between consecutive time stamps values with corresponding positions.
% However, moving sources are not currently handled by the localisation
% algorithm. In this example the same position is then used for start and
% end time stamps.

% Source 1 structure
source1FileName= 'male_s1.wav';
infoAudioFile1 = audioinfo(source1FileName);
sourcesStruct(1) = struct(...
    ...
    'filename',     source1FileName,...                % audio file name : close field recording
    ...
    'ptime',        [0     infoAudioFile1.Duration],...% Time stamps array for corresponding source position (in seconds)
    ...
    'source_xyz',   [2.005 2.005;                      % x : Corresponding source positions (in meters)
                     3.226 3.226;                      % y : Corresponding source positions (in meters)
                     1.193 1.193] ...                  % z : Corresponding source positions (in meters)
    );

% Source 2 structure
source2FileName= 'female_s1.wav';
infoAudioFile2 = audioinfo(source2FileName);
sourcesStruct(2) = struct(...
    ...
    'filename',     source2FileName,...                % audio file name : close field recording
    ...
    'ptime',        [0     infoAudioFile2.Duration],...% Time stamps array for corresponding source position (in seconds)
    ...
    'source_xyz',   [3.405 3.405;                      % x : Corresponding source positions (in meters)
                     3.226 3.226                       % y : Corresponding source positions (in meters)
                     2.183 2.183] ...                  % z : Corresponding source positions (in meters)
    );

if(bDisplayConfiguration)
    display_configuration(roomStruct,sensorsStruct,sourcesStruct)
end

end

function display_configuration(roomStruct,sensorsStruct,sourcesStruct)

figure;

% display room
A = [0 0 0];
B = [roomStruct.room_size(1) 0 0];
C = [0 roomStruct.room_size(2) 0];
D = [0 0 roomStruct.room_size(3)];
E = [0 roomStruct.room_size(2) roomStruct.room_size(3)];
F = [roomStruct.room_size(1) 0 roomStruct.room_size(3)];
G = [roomStruct.room_size(1) roomStruct.room_size(2) 0];
H = [roomStruct.room_size(1) roomStruct.room_size(2) roomStruct.room_size(3)];
P = [A;B;F;H;G;C;A;D;E;H;F;D;E;C;G;B];
plot3(P(:,1),P(:,2),P(:,3),'k')

hold on;

% display sources (red)
for i=1:length(sourcesStruct)
    plot3(sourcesStruct(i).source_xyz(1,1),sourcesStruct(i).source_xyz(2,1),sourcesStruct(i).source_xyz(3,1),'*r');
    text(sourcesStruct(i).source_xyz(1,1),sourcesStruct(i).source_xyz(2,1),sourcesStruct(i).source_xyz(3,1),['S' num2str(i)]);
end

% plot sensor (bleu)
plot3(sensorsStruct.sensor_xyz(:,1),sensorsStruct.sensor_xyz(:,2),sensorsStruct.sensor_xyz(:,3),'*b');

axis equal
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Experiment : room + sources positions (red) + sensors positions (blue)');
clear A B C D E F G H P;
drawnow;
end