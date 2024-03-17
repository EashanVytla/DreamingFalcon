imuclock = [];
imuval = [];
magval = [];
magclock = [];
barroclock = [];
barroval = [];
gpsclock = [];
gpsval = [];
actionclock = [];



%IMU
for i = 1:length(sensorData.x0x2Fmavros0x2Fimu0x2Fdata)
    imuclock(end +1) = ((sensorData.x0x2Fmavros0x2Fimu0x2Fdata{i, 1}.Header.Stamp.Sec)+((sensorData.x0x2Fmavros0x2Fimu0x2Fdata{i, 1}.Header.Stamp.Nsec)/10^9)) - (sensorData.x0x2Fmavros0x2Fimu0x2Fstatic_pressure{1, 1}.Header.Stamp.Sec + ((sensorData.x0x2Fmavros0x2Fimu0x2Fstatic_pressure{1, 1}.Header.Stamp.Nsec)/10^9));
end

for i = 1:length(sensorData.x0x2Fmavros0x2Fimu0x2Fdata)
    imuval(1,i) = sensorData.x0x2Fmavros0x2Fimu0x2Fdata{i, 1}.AngularVelocity.X;
    imuval(2,i) = sensorData.x0x2Fmavros0x2Fimu0x2Fdata{i, 1}.AngularVelocity.Y;
    imuval(3,i) = sensorData.x0x2Fmavros0x2Fimu0x2Fdata{i, 1}.AngularVelocity.Z;
    imuval(4,i) = sensorData.x0x2Fmavros0x2Fimu0x2Fdata{i, 1}.LinearAcceleration.X;
    imuval(5,i) = sensorData.x0x2Fmavros0x2Fimu0x2Fdata{i, 1}.LinearAcceleration.Y;
    imuval(6,i) = sensorData.x0x2Fmavros0x2Fimu0x2Fdata{i, 1}.LinearAcceleration.Z;
    
    for j = 7:21
        imuval(j,i) = -69;
    end
end

imu = [imuclock;imuval];

%Mag

for i = 1:length(sensorData.x0x2Fmavros0x2Fimu0x2Fmag)
    magclock(end +1) = ((sensorData.x0x2Fmavros0x2Fimu0x2Fmag{i, 1}.Header.Stamp.Sec)+((sensorData.x0x2Fmavros0x2Fimu0x2Fmag{i, 1}.Header.Stamp.Nsec)/10^9)) - (sensorData.x0x2Fmavros0x2Fimu0x2Fstatic_pressure{1, 1}.Header.Stamp.Sec + ((sensorData.x0x2Fmavros0x2Fimu0x2Fstatic_pressure{1, 1}.Header.Stamp.Nsec)/10^9));
end

for i = 1:length(sensorData.x0x2Fmavros0x2Fimu0x2Fmag)
    for j = 1:6
        magval(j,i) = -69;
    end

    magval(7,i) = sensorData.x0x2Fmavros0x2Fimu0x2Fmag{i, 1}.MagneticField_.X;
    magval(8,i) = sensorData.x0x2Fmavros0x2Fimu0x2Fmag{i, 1}.MagneticField_.Y;
    magval(9,i) = sensorData.x0x2Fmavros0x2Fimu0x2Fmag{i, 1}.MagneticField_.Z;
    
    for j = 10:21
        magval(j,i) = -69;
    end
end

mag = [magclock;magval];

%Baro
for i = 1:length(sensorData.x0x2Fmavros0x2Fimu0x2Fstatic_pressure)
    barroclock(end +1) = ((sensorData.x0x2Fmavros0x2Fimu0x2Fstatic_pressure{i, 1}.Header.Stamp.Sec) + ((sensorData.x0x2Fmavros0x2Fimu0x2Fstatic_pressure{i, 1}.Header.Stamp.Nsec)/10^9)) - sensorData.x0x2Fmavros0x2Fimu0x2Fstatic_pressure{1, 1}.Header.Stamp.Sec + ((sensorData.x0x2Fmavros0x2Fimu0x2Fstatic_pressure{1, 1}.Header.Stamp.Nsec)/10^9);
end

for i = 1:length(sensorData.x0x2Fmavros0x2Fimu0x2Fstatic_pressure)
    for j = 1:9
        barroval(j,i) = -69;
    end

    barroval(10,i) = sensorData.x0x2Fmavros0x2Fimu0x2Fstatic_pressure{i, 1}.FluidPressure_;
    barroval(11,i) = -69;
    
    for j = 12:21
        barroval(j,i) = -69;
    end
end

barro = [barroclock;barroval];

%GPS 
for i = 1:length(sensorData.x0x2Fmavros0x2Fglobal_position0x2Fraw0x2Ffix)
    gpsclock(end +1) = ((sensorData.x0x2Fmavros0x2Fglobal_position0x2Fraw0x2Ffix{i, 1}.Header.Stamp.Sec) + ((sensorData.x0x2Fmavros0x2Fglobal_position0x2Fraw0x2Ffix{i, 1}.Header.Stamp.Nsec)/10^9)) - sensorData.x0x2Fmavros0x2Fimu0x2Fstatic_pressure{1, 1}.Header.Stamp.Sec + ((sensorData.x0x2Fmavros0x2Fimu0x2Fstatic_pressure{1, 1}.Header.Stamp.Nsec)/10^9);
end

for i = 1:length(sensorData.x0x2Fmavros0x2Fglobal_position0x2Fraw0x2Ffix)
    for j = 1:10
        gpsval(j,i) = -69;
    end
    gpsval(11,i) = sensorData.x0x2Fmavros0x2Fglobal_position0x2Fraw0x2Ffix{i, 1}.Altitude;

    for j = 18:21
        gpsval(j,i) = -69;
    end
end

for i = 1:length(sensorData.x0x2Fmavros0x2Fglobal_position0x2Fraw0x2Fgps_vel)
    gpsval(12,i) = sensorData.x0x2Fmavros0x2Fglobal_position0x2Fraw0x2Fgps_vel{i, 1}.Twist.Linear.X;
    gpsval(13,i) = sensorData.x0x2Fmavros0x2Fglobal_position0x2Fraw0x2Fgps_vel{i, 1}.Twist.Linear.Y;
    gpsval(14,i) = sensorData.x0x2Fmavros0x2Fglobal_position0x2Fraw0x2Fgps_vel{i, 1}.Twist.Linear.Z;
    gpsval(15,i) = sensorData.x0x2Fmavros0x2Fglobal_position0x2Fraw0x2Fgps_vel{i, 1}.Twist.Angular.X;
    gpsval(16,i) = sensorData.x0x2Fmavros0x2Fglobal_position0x2Fraw0x2Fgps_vel{i, 1}.Twist.Angular.Y;
    gpsval(17,i) = sensorData.x0x2Fmavros0x2Fglobal_position0x2Fraw0x2Fgps_vel{i, 1}.Twist.Angular.Z;
end  


gps = [gpsclock;gpsval];

%Action
for i = 1:length(sensorData.x0x2Fmavros0x2Ftarget_actuator_control)
    actionclock(end +1) = ((sensorData.x0x2Fmavros0x2Ftarget_actuator_control{i, 1}.Header.Stamp.Sec) + ((sensorData.x0x2Fmavros0x2Ftarget_actuator_control{i, 1}.Header.Stamp.Nsec)/10^9)) - sensorData.x0x2Fmavros0x2Ftarget_actuator_control{i, 1}.Header.Stamp.Sec + ((sensorData.x0x2Fmavros0x2Ftarget_actuator_control{i, 1}.Header.Stamp.Nsec)/10^9);
end

for i = 1:length(sensorData.x0x2Fmavros0x2Ftarget_actuator_control)
    for j = 1:17
        actval(j,i) = -69;
    end
    
actval(18,i) = sensorData.x0x2Fmavros0x2Ftarget_actuator_control{i, 1}.Controls(1);
actval(19,i) = sensorData.x0x2Fmavros0x2Ftarget_actuator_control{i, 1}.Controls(2);
actval(20,i) = sensorData.x0x2Fmavros0x2Ftarget_actuator_control{i, 1}.Controls(3);
actval(21,i) = sensorData.x0x2Fmavros0x2Ftarget_actuator_control{i, 1}.Controls(4);

end

action = [actionclock;actval];


Bigboi = [imu,mag,barro,gps,action];



Bigboi = transpose(Bigboi);

Bigboi = sortrows(Bigboi, 1);

for i = 2:22
    if Bigboi(1,i) == -69
        Bigboi(1,i) = 0;
    end
end

thingy = length(Bigboi);

for i = 2:thingy
    for j = 2:22
        if Bigboi(i,j) == -69
            Bigboi(i,j) = Bigboi(i-1,j);
        end
    end
end

%imu range
for i = 2:7
    Bigboi(thingy + 1,i) = 25;%max
    Bigboi(thingy + 2,i) = -25;%min
end

%mag range
for i = 8:10
    Bigboi(thingy + 1,i) = 25;%max
    Bigboi(thingy + 2,i) = -25;%min
end

%baro
Bigboi(thingy + 1,11) = 101325;%max
Bigboi(thingy + 2,11) = 0;%min

%gps

Bigboi(thingy + 1,12) = 50;%max alt
Bigboi(thingy + 2,12) = -50;%min alt

for i = 13:18
    Bigboi(thingy + 1,i) = 1;%max
    Bigboi(thingy + 2,i) = -1;%min
end

%Action
for i = 19:22
    Bigboi(thingy + 1,i) = 1;
    Bigboi(thingy + 2,i) = -1;
end

%imu stuff
Bigboi(:,2) = normalize(Bigboi(:,2),'range');
Bigboi(:,3) = normalize(Bigboi(:,3),'range');
Bigboi(:,4) = normalize(Bigboi(:,4),'range');
Bigboi(:,5) = normalize(Bigboi(:,5),'range');
Bigboi(:,6) = normalize(Bigboi(:,6),'range');
Bigboi(:,7) = normalize(Bigboi(:,7),'range');

%mag
Bigboi(:,8) = normalize(Bigboi(:,8),'range');
Bigboi(:,9) = normalize(Bigboi(:,9),'range');
Bigboi(:,10) = normalize(Bigboi(:,10),'range');

%baro
Bigboi(:,11) = normalize(Bigboi(:,11),'range');

%gps
Bigboi(:,12) = normalize(Bigboi(:,12),'range');
Bigboi(:,13) = normalize(Bigboi(:,13),'range');
Bigboi(:,14) = normalize(Bigboi(:,14),'range');
Bigboi(:,15) = normalize(Bigboi(:,15),'range');
Bigboi(:,16) = normalize(Bigboi(:,16),'range');
Bigboi(:,17) = normalize(Bigboi(:,17),'range');
Bigboi(:,18) = normalize(Bigboi(:,18),'range');

%Action
Bigboi(:,19) = normalize(Bigboi(:,19),'range');
Bigboi(:,20) = normalize(Bigboi(:,20),'range');
Bigboi(:,21) = normalize(Bigboi(:,21),'range');
Bigboi(:,22) = normalize(Bigboi(:,22),'range');

ActBoi = [Bigboi(:,1),Bigboi(:,19:22)];
Bigboi = Bigboi(:,1:18);


% Uncomment if Eashan is a bot again
% Bigboi = transpose(Bigboi);
% 
% ActBoi = [Bigboi(1,:);Bigboi(19:22,:)];
% 
% Bigboi = Bigboi(1:18,:);

csvwrite('C:\Users\kbs_s\Documents\GitHub\DreamingFalcon\data\2023-10-13-07-28-08\states.csv', Bigboi);

csvwrite('C:\Users\kbs_s\Documents\GitHub\DreamingFalcon\data\2023-10-13-07-28-08\actions.csv', ActBoi);

% hz = 1/(Bigboi(200,1) - Bigboi(199,1));
% 
% disp(hz);
% 
% imuhz = 1/(imuclock(2) - imuclock(1));
% barrohz = 1/(barroclock(2) - barroclock(1));
% gpshz = 1/(gpsclock(2) - gpsclock(1));

%fprintf('IMU: %f\nBarro: %f\nGps: %f\n',imuhz, barrohz, gpshz)

%disp(sensorData.x0x2Fmavros0x2Fimu0x2Fdata{1, 1}.Header.Stamp.Sec + ((sensorData.x0x2Fmavros0x2Fimu0x2Fdata{1, 1}.Header.Stamp.Nsec)/10^9))

%disp(sensorData.x0x2Fmavros0x2Fimu0x2Fstatic_pressure{1, 1}.Header.Stamp.Sec + ((sensorData.x0x2Fmavros0x2Fimu0x2Fstatic_pressure{1, 1}.Header.Stamp.Nsec)/10^9))

%disp(sensorData.x0x2Fmavros0x2Fglobal_position0x2Fraw0x2Ffix{1, 1}.Header.Stamp.Sec + ((sensorData.x0x2Fmavros0x2Fglobal_position0x2Fraw0x2Ffix{1, 1}.Header.Stamp.Nsec)/10^9))

%figure
%plot(imuclock,imuval)

