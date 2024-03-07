% Specify the path to your bag file
bagFilePath = '2023-10-13-07-28-08.bag';

% Load the bag file
bag = rosbag(bagFilePath);

% Define the topics of interest, including '/mavros_msgs/ActuatorControl'
topics = { '/mavros/imu/data', ...
           '/mavros/imu/mag', ...
           '/mavros/imu/data_raw', ...
           '/mavros/imu/static_pressure', ...
           '/mavros/global_position/raw/fix', ...
           '/mavros/global_position/raw/gps_vel', ...
           '/mavros/rc/out', ...
           '/mavros/target_actuator_control' };

%topics = bag.AvailableTopics;
% Initialize a struct to store the data
sensorData = struct();

% Iterate through each topic
for i = 1:numel(topics)
    % Select messages from the current topic
    msgs = select(bag, 'Topic', topics{i});
    
    % Extract data from the messages and store it in the struct
    sensorData.(genvarname(topics{i})) = readMessages(msgs);
end




% Display the struct containing the sensor data
disp(sensorData);