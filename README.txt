# Visual-Inertial Simultaneous Localization and Mapping (SLAM)
## Visual-Inertial Simultaneous Localization and Mapping (SLAM) Project
Visual-inertial simultaneous localization and mapping (SLAM) is implemented to localize the vehicle and perform landmark mapping of the environment using the extended Kalman filter (EKF) given that the measurements from the vehicle's inertial measurement unit (IMU) and the stereo camera 


## How to Run "Visual-Inertial SLAM"?
1.) Open up the "main.py".

2.) Run the "main.py" code to perform IMU localization via the EKF prediction, landmark mapping via the EKF update, and the visual-inertial SLAM.

3.) It will output the results of IMU localization via the EKF prediction, landmark mapping via the EKF update, and the visual-inertial SLAM.

## Parameters that Can be Changed:
1.) feature_step (Line 13)
- integer type
- Skipped every feature_step number of features.
- Increase the feature step size will increase computational speed but output lesser landmarks.

2.) display_plots (Line 14)
- boolean type
- True -> Display Plot
- False -> Disable the plot display.

## "main.py" Description:
- Load Stereo Camera feature data.
- Perform IMU localization via the EKF prediction step by calling IMU_localization_EKF_predict() function from the "Visual_Inertial_SLAM.py".
- Perform landmark mapping via the EKF update step by calling visual_mapping_EKF_update() function from "Visual_Inertial_SLAM.py".
- Perform visual-inertial SLAM by calling visual_inertial_SLAM() function from "Visual_Inertial_SLAM.py".
- Display all the results of IMU localization via the EKF prediction, landmark mapping via the EKF update, and the visual-inertial SLAM.

## "Visual_Inertial_SLAM.py" Description:
- IMU_localization_EKF_predict() -> The EKF prediction step based on the SE(3) kinematics and the linear and angular velocity measuremetns to estimate the pose T_t is in SE(3) of the IMU over time t.
- visual_mapping_EKF_update() -> Given an EKF with the unknown landmark positions m R^(3 x M) as a state, the EKF update steps are performed after every visual observation z_t in order to keep track of the mean and covariance of m.
- visual_inertial_SLAM() -> Performing Virtual-Inertial SLAM by performing IMU localization via EKF prediction and Land-mapping via EKF Update Step.

## "pr3_utils.py" Description:
- load_data() -> Function to read visual features, IMU measurements and calibration parameters.
- visualize_trajectory_2d() -> Function to visualize the trajectory in 2D.
- skew_sym_mat3() -> Generate 3 x 3 skew-symmetric matrix from a R^3 vector.
- get_u_hat_mat4() -> Generate 4 x 4 generalized velocity from a 3 x 3 skew-symmetric matrix and a R^3 vector.
- get_u_hat_mat6() -> Generate 6 x 6 generalized velocity from two 3 x 3 skew-symmetric matrices.
- get_stereo_camera_calibration_mat() -> Generate stereo camera calibration matrix, M.
- circle_operator() -> Perform circle operator for m in R^4.
- twist() -> Map u = [v, w]^T in R^{6x1} to se(3) in 4 x 4.
- projection() -> Get the projection of a vector, q in R4.
- proj_derivative() -> Compute the derivative of its projection function q
- pixels2world() -> Convert a pixel coordinate from a stereo camera to the world frame.
