imuclock = [];
imuval = [];
magval = [];
magclock = [];
barroclock = [];
barroval = [];
gpsclock = [];
gpsval = [];



%IMU
for i = 1:8208
    imuclock(end +1) = ((sensorData.x0x2Fmavros0x2Fimu0x2Fdata{i, 1}.Header.Stamp.Sec)+((sensorData.x0x2Fmavros0x2Fimu0x2Fdata{i, 1}.Header.Stamp.Nsec)/10^9)) - (sensorData.x0x2Fmavros0x2Fimu0x2Fstatic_pressure{1, 1}.Header.Stamp.Sec + ((sensorData.x0x2Fmavros0x2Fimu0x2Fstatic_pressure{1, 1}.Header.Stamp.Nsec)/10^9));
end

for i = 1:8208
    imuval(1,i) = sensorData.x0x2Fmavros0x2Fimu0x2Fdata{i, 1}.AngularVelocity.X;
    imuval(2,i) = sensorData.x0x2Fmavros0x2Fimu0x2Fdata{i, 1}.AngularVelocity.Y;
    imuval(3,i) = sensorData.x0x2Fmavros0x2Fimu0x2Fdata{i, 1}.AngularVelocity.Z;
    imuval(4,i) = sensorData.x0x2Fmavros0x2Fimu0x2Fdata{i, 1}.LinearAcceleration.X;
    imuval(5,i) = sensorData.x0x2Fmavros0x2Fimu0x2Fdata{i, 1}.LinearAcceleration.Y;
    imuval(6,i) = sensorData.x0x2Fmavros0x2Fimu0x2Fdata{i, 1}.LinearAcceleration.Z;
    imuval(7,i) = -69;
    imuval(8,i) = -69;
    imuval(9,i) = -69;
    imuval(10,i) = -69;
    imuval(11,i) = -69;
    imuval(12,i) = -69;
    imuval(13,i) = -69;
    imuval(14,i) = -69;
    imuval(15,i) = -69;
    imuval(16,i) = -69;
    imuval(17,i) = -69;
end

imu = [imuclock;imuval];

%Mag

for i = 1:8208
    magclock(end +1) = ((sensorData.x0x2Fmavros0x2Fimu0x2Fmag{i, 1}.Header.Stamp.Sec)+((sensorData.x0x2Fmavros0x2Fimu0x2Fmag{i, 1}.Header.Stamp.Nsec)/10^9)) - (sensorData.x0x2Fmavros0x2Fimu0x2Fstatic_pressure{1, 1}.Header.Stamp.Sec + ((sensorData.x0x2Fmavros0x2Fimu0x2Fstatic_pressure{1, 1}.Header.Stamp.Nsec)/10^9));
end

for i = 1:8208
    magval(1,i) = -69;
    magval(2,i) = -69;
    magval(3,i) = -69;
    magval(4,i) = -69;
    magval(5,i) = -69;
    magval(6,i) = -69;
    magval(7,i) = sensorData.x0x2Fmavros0x2Fimu0x2Fmag{i, 1}.MagneticField_.X;
    magval(8,i) = sensorData.x0x2Fmavros0x2Fimu0x2Fmag{i, 1}.MagneticField_.Y;
    magval(9,i) = sensorData.x0x2Fmavros0x2Fimu0x2Fmag{i, 1}.MagneticField_.Z;
    magval(10,i) = -69;
    magval(11,i) = -69;
    magval(12,i) = -69;
    magval(13,i) = -69;
    magval(14,i) = -69;
    magval(15,i) = -69;
    magval(16,i) = -69;
    magval(17,i) = -69;
end

mag = [magclock;magval];

%Baro
for i = 1:3055
    barroclock(end +1) = ((sensorData.x0x2Fmavros0x2Fimu0x2Fstatic_pressure{i, 1}.Header.Stamp.Sec) + ((sensorData.x0x2Fmavros0x2Fimu0x2Fstatic_pressure{i, 1}.Header.Stamp.Nsec)/10^9)) - sensorData.x0x2Fmavros0x2Fimu0x2Fstatic_pressure{1, 1}.Header.Stamp.Sec + ((sensorData.x0x2Fmavros0x2Fimu0x2Fstatic_pressure{1, 1}.Header.Stamp.Nsec)/10^9);
end

for i = 1:3055
    barroval(1,i) = -69;
    barroval(2,i) = -69;
    barroval(3,i) = -69;
    barroval(4,i) = -69;
    barroval(5,i) = -69;
    barroval(6,i) = -69;
    barroval(7,i) = -69;
    barroval(8,i) = -69;
    barroval(9,i) = -69;
    barroval(10,i) = sensorData.x0x2Fmavros0x2Fimu0x2Fstatic_pressure{i, 1}.FluidPressure_;
    barroval(11,i) = -69;
    barroval(12,i) = -69;
    barroval(13,i) = -69;
    barroval(14,i) = -69;
    barroval(15,i) = -69;
    barroval(16,i) = -69;
    barroval(17,i) = -69;
end

barro = [barroclock;barroval];

%GPS 
for i = 1:1283
    gpsclock(end +1) = ((sensorData.x0x2Fmavros0x2Fglobal_position0x2Fraw0x2Ffix{i, 1}.Header.Stamp.Sec) + ((sensorData.x0x2Fmavros0x2Fglobal_position0x2Fraw0x2Ffix{i, 1}.Header.Stamp.Nsec)/10^9)) - sensorData.x0x2Fmavros0x2Fimu0x2Fstatic_pressure{1, 1}.Header.Stamp.Sec + ((sensorData.x0x2Fmavros0x2Fimu0x2Fstatic_pressure{1, 1}.Header.Stamp.Nsec)/10^9);
end

for i = 1:1283
    gpsval(1,i) = -69;
    gpsval(2,i) = -69;
    gpsval(3,i) = -69;
    gpsval(4,i) = -69;
    gpsval(5,i) = -69;
    gpsval(6,i) = -69;
    gpsval(7,i) = -69;
    gpsval(8,i) = -69;
    gpsval(9,i) = -69;
    gpsval(10,i) = -69;
    gpsval(11,i) = sensorData.x0x2Fmavros0x2Fglobal_position0x2Fraw0x2Ffix{i, 1}.Altitude;
    gpsval(12,i) = sensorData.x0x2Fmavros0x2Fglobal_position0x2Fraw0x2Fgps_vel{i, 1}.Twist.Linear.X;
    gpsval(13,i) = sensorData.x0x2Fmavros0x2Fglobal_position0x2Fraw0x2Fgps_vel{i, 1}.Twist.Linear.Y;
    gpsval(14,i) = sensorData.x0x2Fmavros0x2Fglobal_position0x2Fraw0x2Fgps_vel{i, 1}.Twist.Linear.Z;
    gpsval(15,i) = sensorData.x0x2Fmavros0x2Fglobal_position0x2Fraw0x2Fgps_vel{i, 1}.Twist.Angular.X;
    gpsval(16,i) = sensorData.x0x2Fmavros0x2Fglobal_position0x2Fraw0x2Fgps_vel{i, 1}.Twist.Angular.Y;
    gpsval(17,i) = sensorData.x0x2Fmavros0x2Fglobal_position0x2Fraw0x2Fgps_vel{i, 1}.Twist.Angular.Z;
end

gps = [gpsclock;gpsval];


Bigboi = [imu,mag,barro,gps];



Bigboi = transpose(Bigboi);

Bigboi = sortrows(Bigboi, 1);

for i = 2:18
    if Bigboi(1,i) == -69
        Bigboi(1,i) = 0;
    end
end

for i = 2:20754
    for j = 2:18
        if Bigboi(i,j) == -69
            Bigboi(i,j) = Bigboi(i-1,j);
        end
    end
end

%imu range
for i = 2:7
    Bigboi(20755,i) = 25;%max
    Bigboi(20756,i) = -25;%min
end

%mag range
for i = 8:10
    Bigboi(20755,i) = 25;%max
    Bigboi(20756,i) = -25;%min
end

%baro
Bigboi(20755,11) = 101325;%max
Bigboi(20756,11) = 0;%min

%gps

Bigboi(20755,12) = 50;%max alt
Bigboi(20756,12) = -50;%min alt

for i = 13:18
    Bigboi(20755,i) = 1;%max
    Bigboi(20756,i) = -1;%min
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

