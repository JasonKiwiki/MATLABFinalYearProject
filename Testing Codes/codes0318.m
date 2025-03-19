robot = loadrobot('universalUR5', 'DataFormat', 'column');
q = zeros(6,1); % 初始關節角度
J = geometricJacobian(robot, q, 'tool0');
J_pinv = pinv(J);
v_des = [0.1; 0; 0; 0; 0; 0];
q_dot = J_pinv * v_des
