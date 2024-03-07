function rawData = saveData( file, folder)
%% Setup
addpath('')

% Names of messages
globalMessage = '/mavros/global_position/local';
positionMessage = '/mavros/local_position/pose';
velocityMessage = '/mavros/local_position/velocity_body';
dataMessage = '/arm_4/stamped';
commandMessage = '/mavros/rc/in';
gpsMessage = '/mavros/global_position/global';
batteryMessage = '/mavros/battery';
imuMessage = '/mavros/imu/data';
stateMessage = '/mavros/state';
px4FlowMessage = '/mavros/px4flow/raw/optical_flow_rad';

% Make results directory if it does not already exist
% if ~exist( fullfile(folder, 'processed' ), 'dir' )
%     mkdir( fullfile(folder, 'processed' ));
% end

%% Load and save data

% Load bag file
bag = rosbag(fullfile(folder, file));
% if exist(folder, 'dir')
%     folderparts = textscan(folder,'%s','delimiter',{'\','/'}); % splitting folder parts
%     folderparts = folderparts{1};
%     date = folderparts{end};% extracting date from folder name
%     myyear = date(1:4); mymonth = date(5:6); myday = date(7:8);
% 
% end

% state
state = select(bag, 'Topic', stateMessage);
stateMsg = readMessages(state, 'DataFormat','struct');
tState = double( cellfun( @(x)x.Header.Stamp.Sec, stateMsg ) )+double( cellfun( @(x)x.Header.Stamp.Nsec, stateMsg ) )/10^9;

if ~isempty( tState )
    rawData.times.tState = tState - 3600; % adjusting for timezone
    rawData.data.state = double( cellfun( @(x)x.Armed, stateMsg ));
    
    rawData.dependencies.tState = 'state';
else
    rawData.times.tState = nan;
    rawData.data.state = zeros( 2, 0 );
end

% px4 flow
px4flow = select(bag, 'Topic', px4FlowMessage);
px4flowMsg = readMessages(px4flow, 'DataFormat','struct');
tPX4flow = double( cellfun( @(x)x.Header.Stamp.Sec, px4flowMsg ) )+double( cellfun( @(x)x.Header.Stamp.Nsec, px4flowMsg ) )/10^9;

if ~isempty( tPX4flow )
    rawData.times.tPX4flow = tPX4flow - 3600; % adjusting for timezone
    rawData.data.px4flowXYZ = double( [cellfun( @(x)x.IntegratedX, px4flowMsg ),...
        cellfun( @(x)x.IntegratedY, px4flowMsg ),...
        cellfun( @(x)x.IntegratedXgyro, px4flowMsg ),...
        cellfun( @(x)x.IntegratedYgyro, px4flowMsg ),...
        cellfun( @(x)x.IntegratedZgyro, px4flowMsg ) ]);
    rawData.data.px4flowDist = double(cellfun( @(x)x.Distance, px4flowMsg));
    rawData.data.px4flowQual = double(cellfun( @(x)x.Quality, px4flowMsg ));
    
    rawData.dependencies.tPX4flow = {'px4flowXYZ', 'px4flowQual'};
else
    rawData.times.tPX4flow = nan;
    rawData.data.px4flow = zeros( 2, 0 );
end

% GPS Location
GPS = select( bag, 'Topic', gpsMessage );
gpsMes = readMessages( GPS, 'dataformat', 'struct' );
% extracting times from GPS data
tGPS = double( cellfun( @(x)x.Header.Stamp.Sec, gpsMes ) )+double( cellfun( @(x)x.Header.Stamp.Nsec, gpsMes ) )/10^9;

if ~isempty( tGPS )
    rawData.times.tGPS = tGPS - 3600; % adjusting for timezone
    rawData.data.GPS = double( [cellfun( @(x)x.Latitude, gpsMes ),...
        cellfun( @(x)x.Longitude, gpsMes ),...
        cellfun( @(x)x.Altitude, gpsMes )] );
    
    dateTimes = datetime(tGPS,'ConvertFrom','posixtime');
    startTime = tGPS(1);
    endTime = tGPS(end);
    
    rawData.dependencies.tGPS = 'GPS';
else
    rawData.times.tGPS = nan;
    rawData.data.GPS = zeros( 2, 3 );
    dateTimes = 0;
    startTime = 0;
    endTime = 0;
end

% Get data from position message
Pos = select( bag, 'Topic', positionMessage );
PosMes = readMessages( Pos, 'dataformat', 'struct' );
tPos = double( cellfun( @(x)x.Header.Stamp.Sec, PosMes ) )+double( cellfun( @(x)x.Header.Stamp.Nsec, PosMes ) )/10^9;
if ~isempty( tPos )
    rawData.times.tPos = tPos - 3600; % adjusting for timezone
    rawData.data.pos = [cellfun( @(x)x.Pose.Position.X, PosMes ),...
        cellfun( @(x)x.Pose.Position.Y, PosMes ),...
        cellfun( @(x)x.Pose.Position.Z, PosMes )];

    rawData.data.quat = [cellfun( @(x)x.Pose.Orientation.W, PosMes ),...
        cellfun( @(x)x.Pose.Orientation.X, PosMes ),...
        cellfun( @(x)x.Pose.Orientation.Y, PosMes ),...
        cellfun( @(x)x.Pose.Orientation.Z, PosMes )];

    rawData.dependencies.tPos = {'pos','quat'};
else
    rawData.times.tPos = nan;
    rawData.data.pos = zeros( 2, 3 );
    rawData.data.quat = zeros( 2, 4 );
    rawData.data.quat(:,1) = 1;
end

% Get data from position message
Vel = select( bag, 'Topic', velocityMessage );
VelMes = readMessages( Vel, 'dataformat', 'struct' );
tVel = double( cellfun( @(x)x.Header.Stamp.Sec, VelMes ) )+double( cellfun( @(x)x.Header.Stamp.Nsec, VelMes ) )/10^9;
if ~isempty( tVel )
    rawData.times.tVel = tVel - 3600; % adjusting for timezone
    rawData.data.linVel = [cellfun( @(x)x.Twist.Linear.X, VelMes ),...
        cellfun( @(x)x.Twist.Linear.Y, VelMes ),...
        cellfun( @(x)x.Twist.Linear.Z, VelMes )];

    rawData.data.angVel = [cellfun( @(x)x.Twist.Angular.X, VelMes ),...
        cellfun( @(x)x.Twist.Angular.Y, VelMes ),...
        cellfun( @(x)x.Twist.Angular.Z, VelMes )];

    rawData.dependencies.tVel = {'linVel','angVel'};
else
    rawData.times.tVel = nan;
    rawData.data.linVel = zeros( 2, 4 );
    rawData.data.angVel = zeros( 2, 4 );

end

% Get data from arms
Data = select( bag, 'Topic', dataMessage );
DataMes = readMessages( Data, 'dataformat', 'struct' );

if ~isempty(DataMes)
    tData = double( cellfun( @(x)x.Header.Stamp.Sec, DataMes ) )+double( cellfun( @(x)x.Header.Stamp.Nsec, DataMes ) )/10^9;
else
    dataMessage = '/arm_4';
    Data = select( bag, 'Topic', dataMessage );
    DataMes = readMessages( Data, 'dataformat', 'struct' );

    tData = double(Data.MessageList.Time);
end

if ~isempty( tData )
    rawData.times.tData = tData - 3600; % adjusting for timezone
    rawData.data.Voltage = double( [cellfun( @(x)x.V1, DataMes ),...
        cellfun( @(x)x.V2, DataMes ),...
        cellfun( @(x)x.V3, DataMes ),...
        cellfun( @(x)x.V4, DataMes )] );

    rawData.data.Current = double( [cellfun( @(x)x.C1, DataMes ),...
        cellfun( @(x)x.C2, DataMes ),...
        cellfun( @(x)x.C3, DataMes ),...
        cellfun( @(x)x.C4, DataMes )] );

    rawData.data.Thrust = double( [cellfun( @(x)x.THRUST1, DataMes ),...
        cellfun( @(x)x.THRUST2, DataMes ),...
        cellfun( @(x)x.THRUST3, DataMes ),...
        cellfun( @(x)x.THRUST4, DataMes )] );

    rawData.data.Strain = double( [cellfun( @(x)x.STRAIN1, DataMes ),...
        cellfun( @(x)x.STRAIN2, DataMes ),...
        cellfun( @(x)x.STRAIN3, DataMes ),...
        cellfun( @(x)x.STRAIN4, DataMes )] );

    rawData.data.Speed = double( [cellfun( @(x)x.RPM1, DataMes ),...
        cellfun( @(x)x.RPM2, DataMes ),...
        cellfun( @(x)x.RPM3, DataMes ),...
        cellfun( @(x)x.RPM4, DataMes )] );

    rawData.dependencies.tData = {'Voltage','Current','Thrust','Strain','Speed'};
else
    rawData.times.tData = nan;
    rawData.data.Voltage = zeros( 2, 4 );
    rawData.data.Current = zeros( 2, 4 );
    rawData.data.Thrust = zeros( 2, 4 );
    rawData.data.Strain = zeros( 2, 4 );
    rawData.data.Speed = zeros( 2, 4 );

end

% Get data from stick
Command = select( bag, 'Topic', commandMessage );
commandMes = readMessages( Command, 'dataformat', 'struct' );

tCommand = double( cellfun( @(x)x.Header.Stamp.Sec, commandMes ) )+double( cellfun( @(x)x.Header.Stamp.Nsec, commandMes ) )/10^9;
if ~isempty( tCommand )
    channels = cellfun( @(x)x.Channels, commandMes, 'UniformOutput', false );
    indx = cellfun( @isempty, channels );

    channels(indx) = [];
    tCommand(indx) = [];

    rawData.times.tCommand = tCommand - 3600; % adjusting for timezone
    Command = double( cell2mat( channels ) );
    rawData.data.Command = reshape( Command, length( commandMes{1}.Channels ), [] )';
    rawData.data.armed = rawData.data.Command(:,8) - rawData.data.Command(1,8);

    rawData.dependencies.tCommand = 'armed';
else
    rawData.times.tCommand = nan;
    rawData.data.Command = 0;

end

% Pixhawk battery data
Battery = select( bag, 'Topic', batteryMessage );
BatteryMes = readMessages( Battery, 'dataformat', 'struct' );

tBattery = double( cellfun( @(x)x.Header.Stamp.Sec, BatteryMes ) )+double( cellfun( @(x)x.Header.Stamp.Nsec, BatteryMes ) )/10^9;
if ~isempty( tBattery )
    rawData.times.tBattery = tBattery - 3600; % adjusting for timezone
    rawData.data.pixhawkVoltage = double( cellfun( @(x)x.Voltage, BatteryMes ) );
    rawData.data.pixhawkCurrent = double( cellfun( @(x)x.Current, BatteryMes ) );

    rawData.dependencies.tBattery = {'pixhawkVoltage','pixhawkCurrent'};
else
    rawData.times.tBattery = nan;
    rawData.data.pixhawkVoltage = zeros( 2, 1 );
    rawData.data.pixhawkCurrent = zeros( 2, 1 );

end

% Global position and velocity data
globalPos = select( bag, 'Topic', globalMessage );
globalPosMes = readMessages( globalPos, 'dataformat', 'struct' );

tGlobal = double( cellfun( @(x)x.Header.Stamp.Sec, globalPosMes ) )+double( cellfun( @(x)x.Header.Stamp.Nsec, globalPosMes ) )/10^9;
if ~isempty( tGlobal )
    rawData.times.tGlobal = tGlobal - 3600; % adjusting for timezone
    rawData.data.posGlobal = [cellfun( @(x)x.Pose.Pose.Position.X, globalPosMes ),...
        cellfun( @(x)x.Pose.Pose.Position.Y, globalPosMes ),...
        cellfun( @(x)x.Pose.Pose.Position.Z, globalPosMes )];

    rawData.data.quatGlobal = [cellfun( @(x)x.Pose.Pose.Orientation.W, globalPosMes ),...
        cellfun( @(x)x.Pose.Pose.Orientation.X, globalPosMes ),...
        cellfun( @(x)x.Pose.Pose.Orientation.Y, globalPosMes ),...
        cellfun( @(x)x.Pose.Pose.Orientation.Z, globalPosMes )];

    rawData.data.linVelGlobal = [cellfun( @(x)x.Twist.Twist.Linear.X, globalPosMes ),...
        cellfun( @(x)x.Twist.Twist.Linear.Y, globalPosMes ),...
        cellfun( @(x)x.Twist.Twist.Linear.Z, globalPosMes )];

    rawData.data.angVelGlobal = [cellfun( @(x)x.Twist.Twist.Angular.X, globalPosMes ),...
        cellfun( @(x)x.Twist.Twist.Angular.Y, globalPosMes ),...
        cellfun( @(x)x.Twist.Twist.Angular.Z, globalPosMes )];

    rawData.dependencies.tGlobal = {'posGlobal','quatGlobal','linVelGlobal','angVelGlobal'};
else
    rawData.times.tGlobal = nan;
    rawData.data.posGlobal = zeros( 2, 4 );
    rawData.data.quatGlobal = zeros( 2, 4 );
    rawData.data.quatGlobal(:,1) = 1;
    rawData.data.linVelGlobal = zeros( 2, 4 );
    rawData.data.angVelGlobal = zeros( 2, 4 );

end

% Get acceleration data
IMU = select( bag, 'Topic', imuMessage );
imuMes = readMessages( IMU, 'dataformat', 'struct' );

tIMU = double( cellfun( @(x)x.Header.Stamp.Sec, imuMes ) )+double( cellfun( @(x)x.Header.Stamp.Nsec, imuMes ) )/10^9;
if ~isempty( tIMU )
    rawData.times.tIMU = tIMU - 3600; % adjusting for timezone
    rawData.data.linAccel = [cellfun( @(x) x.LinearAcceleration.X, imuMes ),...
        cellfun( @(x) x.LinearAcceleration.Y, imuMes ),...
        cellfun( @(x) x.LinearAcceleration.Z, imuMes ) ];

    rawData.dependencies.tIMU = 'linAccel';
else
    rawData.times.tIMU = nan;
    rawData.data.linAccel = zeros( 2, 3 );

end


if startTime == 0 && endTime == 0 && dateTimes == 0 % if GPS data is unavailable...
    dateTimes = datetime(tData,'ConvertFrom','posixtime');
    startTime = tData(1);
    endTime = tData(end);
end

tAir = [startTime,endTime];
try
    % Air Properties
    air = getAirDataOgimet( 'OSU', mean( tAir ) );
catch ME
    air = [];
    warning(ME.message);

end

% Save Data
% mytime = string(timeofday(dateTimes(1)));
% mytime_split = split(mytime,':');
% matname = strcat(myyear,'-',mymonth,'-',myday,'-',mytime_split(1),'-',mytime_split(2),'-',mytime_split(3));
matname = strcat(file(1:19),".mat");
save( fullfile(folder, matname),...
    'air', 'rawData');
end