%% 1. Import or Load the Robot Model

robot = loadrobot("universalUR5", "DataFormat", "row");
% Display general information about the rigidBodyTree
showdetails(robot);

%% 2. Display the Robot in a Static Configuration
% Define a 6-DOF joint configuration. 

q0 = zeros(1,6);  % All joints at zero angle
figure('Name','Static Robot View','NumberTitle','off');
show(robot, q0, "Frames","on", "Visuals","on");
title("UR5 at q = [0 0 0 0 0 0]");

%% 3. Forward Kinematics
% getTransform() to find the end-effector pose for a given joint configuration.

qTest = [0, -pi/2, pi/3, 0, pi/4, 0];
tform = getTransform(robot, qTest, "tool0");  % "tool0" is the name of the end effector link
disp("Forward Kinematics Result (End-Effector Pose):");
disp(tform);

%% 4. Inverse Kinematics
% Create an Inverse Kinematics (IK) solver object using the robot model.

ik = inverseKinematics("RigidBodyTree", robot);
% Weights for position and orientation ([x y z roll pitch yaw])
weights = [1 1 1 1 1 1];
% Initial guess for the solver - typically the robot's home configuration
initialguess = homeConfiguration(robot);

% Define a target pose for the end effector
targetPose = trvec2tform([0.3, 0.2, 0.5]);  % position only, no rotation

% Perform the IK solve
[configSol, solInfo] = ik("tool0", targetPose, weights, initialguess);
disp("Inverse Kinematics Solution (Joint Angles):");
disp(configSol);
disp("Solution Info:");
disp(solInfo);

%% 5. Simple Animation / Trajectory Demonstration
% Move the robot through a sine wave motion for the first joint, while keeping the others fixed, and visualize it in a loop.

figure('Name','Robot Animation','NumberTitle','off');
for t = linspace(0, 2*pi, 50)
    % Generate a joint configuration - only the first joint moves over time
    qAnim = [0.5*sin(t), -pi/4, pi/6, 0, pi/6, 0];
    
    % Show the robot with the updated joint angles
    show(robot, qAnim, "Frames","off", "Visuals","on", "PreservePlot", false);
    title("Sine Wave Animation on the First Joint");
    drawnow;  % Refresh the figure
end

%% 6. Using IK in a Loop for a Path
% Move the end-effector in a circle in the XY-plane, and solve for joint angles at each step.

figure('Name','IK Circular Trajectory','NumberTitle','off');
initialguess = homeConfiguration(robot);  % Reset initial guess for each iteration
for theta = linspace(0, 2*pi, 60)
    % Define a circular path for the end-effector: center (0.4, 0.2), radius 0.1
    xPos = 0.4 + 0.1*cos(theta);
    yPos = 0.2 + 0.1*sin(theta);
    zPos = 0.3;  % constant z
    
    % Create the target pose
    targetTform = trvec2tform([xPos, yPos, zPos]);
    
    % Solve IK
    [configSol, ~] = ik("tool0", targetTform, weights, initialguess);
    
    % Convert the solution to a numeric row vector for all joints
    qNow = zeros(1,6);
    for j = 1:6
        qNow(j) = configSol(j).JointPosition;
    end
    
    % Update the initial guess to help solver convergence
    initialguess = configSol;
    
    % Display the robot at the new configuration
    show(robot, qNow, "PreservePlot", false, "Frames","off", "Visuals","on");
    title("IK Circular Motion Demo");
    drawnow;
end
