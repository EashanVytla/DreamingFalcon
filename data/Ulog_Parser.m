imuclock = [];
imuval = [];
magval = [];
magclock = [];
barroclock = [];
barroval = [];
gpsclock = [];
gpsval = [];
actionclock = [];



%Sensors
for i = 1:length(ans.sensor_accel.A.timestamp_sample)
    imuclock(end +1) = seconds(ans.sensor_accel.A.timestamp_sample(i));
end

for i = 1:length(ans.sensor_accel.A.timestamp_sample)
    imuval(1,i) = ans.sensor_accel.A.x(i);
    imuval(2,i) = ans.sensor_accel.A.y(i);
    imuval(3,i) = ans.sensor_accel.A.z(i);
    imuval(4,i) = ans.sensor_gyro.A.x(i);
    imuval(5,i) = ans.sensor_gyro.A.y(i);
    imuval(6,i) = ans.sensor_gyro.A.z(i);
    imuval(7,i) = ans.sensor_mag.A.x(i);
    imuval(8,i) = ans.sensor_mag.A.y(i);
    imuval(9,i) = ans.sensor_mag.A.z(i);
    imuval(10,i) = ans.sensor_baro.A.pressure(i);
    imuval(11,i) = ans.sensor_baro.B.pressure(i);

    for j = 12:23
        imuval(j,i) = -69;
    end
end

imu = [imuclock;imuval];


%Action
for i = 1:length(ans.actuator_motors.A.control)
    actionclock(end +1) = seconds(ans.actuator_motors.A.timestamp(i));
end

for i = 1:length(ans.actuator_motors.A.control)
    for j = 1:11
        actval(j,i) = -69;
    end
    
actval(12,i) = ans.actuator_motors.A.control(i,1);
actval(13,i) = ans.actuator_motors.A.control(i,2);
actval(14,i) = ans.actuator_motors.A.control(i,3);
actval(15,i) = ans.actuator_motors.A.control(i,4);
actval(16,i) = ans.vehicle_attitude.A.q(i,1);
actval(17,i) = ans.vehicle_attitude.A.q(i,2);
actval(18,i) = ans.vehicle_attitude.A.q(i,3);
actval(19,i) = ans.vehicle_attitude.A.q(i,4);
actval(20,i) = ans.vehicle_attitude_setpoint.A.q_d(i,1);
actval(21,i) = ans.vehicle_attitude_setpoint.A.q_d(i,2);
actval(22,i) = ans.vehicle_attitude_setpoint.A.q_d(i,3);
actval(23,i) = ans.vehicle_attitude_setpoint.A.q_d(i,4);

end

action = [actionclock;actval];


Bigboi = [imu,action];



Bigboi = transpose(Bigboi);

Bigboi = sortrows(Bigboi, 1);

for i = 2:24
    if Bigboi(1,i) == -69
        Bigboi(1,i) = 0;
    end
end

thingy = length(Bigboi);

for i = 2:thingy
    for j = 2:24
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


siu = 0;

for i = 2:thingy
    if Bigboi(i-1,1) == Bigboi(i,1)
        siu = siu + 1;
    end
end


%------------------Normalizing----------
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
Bigboi(:,11) = normalize(Bigboi(:,9),'range');
Bigboi(:,12) = normalize(Bigboi(:,10),'range');

idkboi = Bigboi(2,1);
for i = 1:length(Bigboi)
    Bigboi(i,1) = Bigboi(i,1) - idkboi;
end


Bigboi(:,1) = floor(Bigboi(:,1) * 10^1) / 10^1;

utv = unique(Bigboi(:,1));

otherboi = zeros(1,24);

for i = 4:thingy
    if Bigboi(i-1,1) == Bigboi(i,1)
        otherboi(end,:) = Bigboi(i,:);
    else
        otherboi(end + 1,:) = Bigboi(i,:);
    end
end


%------------
x = 0;

%otherboi = otherboi(1000:end-1000,:);


total = length(otherboi);

train = floor(.7*total);

val = floor(.2 * total);

test = floor(.1 * total);




ActBoi_train = [otherboi(1:train,1),otherboi(1:train,13:16)];
Betterboi_train = otherboi(1:train,1:12);

ActBoi_val = [otherboi(train:train + val,1),otherboi(train:train + val,13:16)];
Betterboi_val = otherboi(train:train + val,1:12);

ActBoi_test = [otherboi(train + val:train + val + test,1),otherboi(train + val:train + val + test,13:16)];
Betterboi_test = otherboi(train + val:train + val + test,1:12);

rewards_set = [];


for i = 1:total
    rewards_set(end+1) = (1 - abs(otherboi(i,21) - otherboi(i,17))) + (1 - abs(otherboi(i,22) - otherboi(i,18))) + (1 - abs(otherboi(i,23) - otherboi(i,19))) + (1 - abs(otherboi(i,24) - otherboi(i,20)));
end



train = floor(.7*total);

val = floor(.2 * total);

test = floor(.1 * total);

rewards_train = [rewards_set(1:train)];


rewards_val = [rewards_set(train:train + val)];


rewards_test = [rewards_set(train + val:train + val + test)];

rewards_train = transpose(rewards_train);
rewards_val = transpose(rewards_val);
rewards_test = transpose(rewards_test);


% Uncomment if Eashan is a bot again
% Bigboi = transpose(Bigboi);
% 
% ActBoi = [Bigboi(1,:);Bigboi(19:22,:)];
% 
% Bigboi = Bigboi(1:18,:);

csvwrite('C:\Users\kbs_s\Documents\GitHub\DreamingFalcon\data\Mixed\states_train.csv', Betterboi_train);

csvwrite('C:\Users\kbs_s\Documents\GitHub\DreamingFalcon\data\Mixed\actions_train.csv', ActBoi_train);

csvwrite('C:\Users\kbs_s\Documents\GitHub\DreamingFalcon\data\Mixed\states_val.csv', Betterboi_val);

csvwrite('C:\Users\kbs_s\Documents\GitHub\DreamingFalcon\data\Mixed\actions_val.csv', ActBoi_val);

csvwrite('C:\Users\kbs_s\Documents\GitHub\DreamingFalcon\data\Mixed\states_test.csv', Betterboi_test);

csvwrite('C:\Users\kbs_s\Documents\GitHub\DreamingFalcon\data\Mixed\actions_test.csv', ActBoi_test);

csvwrite('C:\Users\kbs_s\Documents\GitHub\DreamingFalcon\data\Mixed\rewards_train.csv', rewards_train);

csvwrite('C:\Users\kbs_s\Documents\GitHub\DreamingFalcon\data\Mixed\rewards_val.csv', rewards_val);

csvwrite('C:\Users\kbs_s\Documents\GitHub\DreamingFalcon\data\Mixed\rewards_test.csv', rewards_test);


% hz = 1/(Bigboi(200,1) - Bigboi(199,1));
% 
% disp(hz);
% 
 % imuhz = 1/(imuclock(2000) - imuclock(1999));
 % barrohz = 1/(barroclock(1999) - barroclock(1998));
 % gpshz = 1/(gpsclock(2000) - gpsclock(1999));

%fprintf('IMU: %f\nBarro: %f\nGps: %f\n',imuhz, barrohz, gpshz)

%disp(sensorData.x0x2Fmavros0x2Fimu0x2Fdata{1, 1}.Header.Stamp.Sec + ((sensorData.x0x2Fmavros0x2Fimu0x2Fdata{1, 1}.Header.Stamp.Nsec)/10^9))

%disp(sensorData.x0x2Fmavros0x2Fimu0x2Fstatic_pressure{1, 1}.Header.Stamp.Sec + ((sensorData.x0x2Fmavros0x2Fimu0x2Fstatic_pressure{1, 1}.Header.Stamp.Nsec)/10^9))

%disp(sensorData.x0x2Fmavros0x2Fglobal_position0x2Fraw0x2Ffix{1, 1}.Header.Stamp.Sec + ((sensorData.x0x2Fmavros0x2Fglobal_position0x2Fraw0x2Ffix{1, 1}.Header.Stamp.Nsec)/10^9))

%figure
%plot(imuclock,imuval)

