%Backup Test_Comparison

normoutput = csvread('testSequenceOutput4-9-2.csv');


output = normoutput * 50 - 25;

un_norm_data = Betterboi_test * 50 - 25;

figure


%IMU
plot(output(:,1),'r','DisplayName','X');
hold on
plot(output(:,2),'g','DisplayName','Y');
hold on
plot(output(:,3),'b','DisplayName','Z');
title('Model Predictions IMU')
legend

ylabel('m/s^2')
xlabel('s/10')

figure
plot(un_norm_data(:,2),'r','DisplayName','X');
hold on
plot(un_norm_data(:,3),'g','DisplayName','Y');
hold on
plot(un_norm_data(:,4),'b','DisplayName','Z');
title('Sim Test Data IMU')

legend

ylabel('m/s^2')
xlabel('s/10')

%Gyro
figure
plot(output(:,4),'r','DisplayName','X');
hold on
plot(output(:,5),'g','DisplayName','Y');
hold on
plot(output(:,6),'b','DisplayName','Z');
title('Model Predictions Gyro')
legend

ylabel('rad/s^2')
xlabel('s/10')

figure
plot(un_norm_data(:,5),'r','DisplayName','X');
hold on
plot(un_norm_data(:,6),'g','DisplayName','Y');
hold on
plot(un_norm_data(:,7),'b','DisplayName','Z');
title('Sim Test Data Gyro')

legend

ylabel('rad/s^2')
xlabel('s/10')

%Mag
figure
plot(output(:,7),'r','DisplayName','X');
hold on
plot(output(:,8),'g','DisplayName','Y');
hold on
plot(output(:,9),'b','DisplayName','Z');
title('Model Predictions Magnetometer')
legend

ylabel('gauss')
xlabel('s/10')

figure
plot(un_norm_data(:,8),'r','DisplayName','X');
hold on
plot(un_norm_data(:,9),'g','DisplayName','Y');
hold on
plot(un_norm_data(:,10),'b','DisplayName','Z');
title('Sim Test Data Magnetometer')

legend

ylabel('gauss')
xlabel('s/10')
