import numpy as np
from scipy.linalg import expm
from pr3_utils import *
from tqdm import tqdm

def IMU_localization_EKF_predict(t, linear_velocity, angular_velocity, motion_noise_var = 1):
    '''
    The EKF prediction step based on the SE(3) kinematics and the linear and angular 
    velocity measuremetns to estimate the pose T_t is in SE(3) of the IMU over time t.

    :Input (Type)   : 1.) t (float)                             - Timestamps
                      2.) linear_velocity (np.ndarray)          - Linear Velocity
                      3.) angular_velocity (np.ndarray)         - Angular Velociry
                      4.) motion_noise_var (float)              - Motion Noise Variance
    :Output(Type)   : 1.) vehicle_poses_imu2world (np.ndarray)  - Vehicle poses in world frame
                      2.) vehicle_poses_world2imu (np.ndarray)  - Vehicle poses in IMU frame
    '''
    # Initialize mean and covariance matrices
    mew_mat = np.eye(4)
    cov_mat = np.eye(6)
    # Initialize vehicle poses
    vehicle_poses_world2imu = np.zeros((4, 4, t.shape[1]))
    vehicle_poses_world2imu[:,:,0] = mew_mat
    vehicle_poses_imu2world = np.zeros((4, 4, t.shape[1]))
    vehicle_poses_imu2world[:,:,0] = np.linalg.inv(mew_mat)

    # Determine the change in time stamps, tau.
    tau = t[:, 1:] - t[:,:-1]
    # Discard the first linear and angular velocities
    lin_vel = linear_velocity[:,1:]
    ang_vel= angular_velocity[:,1:]

    # Start EKF Prediction Step
    for idx in tqdm(range (tau.shape[1])):

        # Calculate current tau, linear velocity, and angular velocity
        tau_curr = tau[:, idx]
        lin_vel_curr = lin_vel[:, idx]
        ang_vel_curr = ang_vel[:, idx]

        # Generate linear and angular velocities skew-symmetric matrix
        lin_vel_hat_mat = skew_sym_mat3(lin_vel_curr)
        ang_vel_hat_mat = skew_sym_mat3(ang_vel_curr)

        # Compute 2 x 3 control, u matrix (Don't need it)
        #u = np.vstack((lin_vel_curr.reshape(1, -1), 
        #                ang_vel_curr.reshape(1, -1)))

        # Compute 4 x 4 control hat matrix
        u_hat4 = get_u_hat_mat4(ang_vel_hat_mat, lin_vel_curr)

        # Compute 6 x 6 control hat matrix
        u_hat6 = get_u_hat_mat6(ang_vel_hat_mat, lin_vel_hat_mat)

        ## Predict vehicle pose with EKF Prediction Step
        # Generate random Gaussian motion noise
        motion_noise_W = np.diag(np.random.normal(0, np.sqrt(motion_noise_var), 6))
        # Calculate the predicted mew and cov matrices
        mew_mat = np.dot(expm(-tau_curr * u_hat4), mew_mat)
        cov_mat_exp = expm(-tau_curr * u_hat6)
        cov_mat = np.dot(np.dot(cov_mat_exp, cov_mat), np.transpose(cov_mat_exp)) + motion_noise_W
        # Get vehicle pose in world frame
        vehicle_poses_imu2world[:,:,idx + 1] = np.linalg.inv(mew_mat)
        # Get vehicle pose in IMU frame
        vehicle_poses_world2imu[:,:,idx + 1] = mew_mat
               
    return vehicle_poses_imu2world, vehicle_poses_world2imu

def visual_mapping_EKF_update(features, feature_step, vehicle_poses_world2imu, K, b, imu_T_cam, observation_noise_var = 100):
    '''
    Implement an EKF with the unknown landmark positions m R^(3 x M) as a state. Perform EKF update steps
    after every visual observation z_t in order to keep track of the mean and covariance of m.

    :Input (Type)   : 1.) features (np.ndarray)                 - Features from stereo images
                      2.) feature_step (int)                    - Feature step
                      2.) vehicle_poses_world2imu (np.ndarray)  - Vehicle poses in IMU frame
                      3.) K (np.ndarray)                        - Stereo Camera Intrinsic Matrix, K
                      4.) b (float)                             - Stereo Camera Baseline, b [meters]
                      5.) imu_T_cam (np.ndarray)                - Transformation Matrix from IMU to Camera
                      6.) observation_noise_var (float)         - Sensor noise variance
    :Output(Type)   : 1.) mew_landmarks (np.ndarray)            - Stereo Camera Landmarks (xy-coordinates) in world frame
    '''
    ## Pre-processing Input Data
    #----------------------------
    # Get the Stereo Camera Calibration Matrix, M
    M_mat = get_stereo_camera_calibration_mat(K, b)
    # Get Smaller Number of Features
    lesser_features = features[:, 1:features.shape[1]:feature_step, :]
    # Number of Features, M
    num_features = lesser_features.shape[1]
    # Dilation matrix
    D = np.vstack((np.eye(3), np.zeros((1, 3)))) # 4 x 3 
    D_mat = np.tile(D,(num_features, num_features)) # 4M x 3M
    # Define non-observable z
    z_empty = np.array([-1, -1, -1, -1]) 
    # Define projection matrix
    P_mat = np.hstack((np.eye(3), np.zeros((3, 1)))) # 3 x 4 
    # Get the transformation matrix from optical frame to IMU
    cam_T_imu = np.linalg.inv(imu_T_cam)

    ## Initialization Stage
    #----------------------
    # Initialize Landmark mean matrix or Landmark poses in homogeneous coordinates (4M x 1) instead of (3M x 1)
    mew_landmarks = np.zeros((4 * num_features, 1))
    # Initialize Joint Covariance for both landmark and IMU pose ((3M) x (3M))
    cov = np.eye(3 * num_features)
    # Intialize the landmarks position that check if the feature is initialized.
    mew_init_status = np.zeros(num_features)
    
    ## Start EKF Update Step for Landmark Mapping
    for idx in tqdm(range (0, lesser_features.shape[2] - 1)):

        ## EKF Update Step for Landmark Mapping
        #----------------------------------------------------
        # Get the IMU pose, T_t (follow the notation from notes) if not, my head is going to explode. too confusing...
        T_t = vehicle_poses_world2imu[:,:,idx + 1]
        # Extract an array of features at time t, f_t (lazy already, make the notation shorter)
        f_t = lesser_features[:, :, idx]
        # Generate Gaussian observation noise
        observation_noise_v = np.random.normal(0, np.sqrt(observation_noise_var)) # noise

        # Iterate through the landmarks at time t
        for j in range(0, num_features):

            # Initialize Jacobian Observation Matrix, Hijs as a list including the landmark with indices. [Hij_landmark, j]
            # Note: the Hij's j and j here are different. Don't messed up!
            Hijs_joint = []
            # Initialize observation, z_tplus1 as an empty array
            z_tplus1 = np.array([])
            # Initialize observation with a hat as an empty array
            tilde_z_tplus1 = np.array([])

            # If there's are observable features
            if (f_t[:,j] != z_empty).all():

                # If the landmark is not initialized
                if mew_init_status[j] == 0: 

                    # Mark as initialize
                    mew_init_status[j] = 1
                    # Initialze the landmark mew
                    mew_landmarks[4*j : 4*(j+1), :]  = pixels2world(f_t[:,j], K,b, cam_T_imu, T_t)
                    # Initialize the landmark covariance
                    cov[3*j : 3*(j+1), 3*j : 3*(j+1)] = np.eye(3) * observation_noise_var

                # Update the initialzed mew_landmark and cov_landmark
                else:

                    # Get current landmarks mean
                    curr_mew_landmark = mew_landmarks[4*j : 4*(j+1), :]
                    # Find q_landmark
                    q_landmark = cam_T_imu @ T_t @ curr_mew_landmark
                    # Find pi_q_landmark
                    pi_q_landmark = projection(q_landmark)
                    # Determine the derivative of the projection function dpi(q)/dq
                    dpi_dq_landmark = proj_derivative(q_landmark)
                    # Find z_t+1 
                    z_tp1 = M_mat @ pi_q_landmark + observation_noise_v
                    # Find tilde_z_t+1 
                    tilde_z_tp1 = M_mat @ pi_q_landmark
                    # Store z_t+1
                    z_tplus1 = np.concatenate((z_tplus1, z_tp1), axis=None)
                    # Store tilde_z_t+1
                    tilde_z_tplus1 = np.concatenate((tilde_z_tplus1, tilde_z_tp1), axis=None)
                    # Calculate Hij_landmark
                    Hij_landmark = M_mat @ dpi_dq_landmark @ cam_T_imu @ T_t @ P_mat.T
                    # Append to Hijs list
                    Hijs_joint.append((Hij_landmark, j))

        # Number of observable features, N_t
        N_t = len(Hijs_joint)

        # Reshape z_tplus1 and tilde_z_tplus1
        z_tplus1 = z_tplus1.reshape((4*N_t, 1))
        tilde_z_tplus1 = tilde_z_tplus1.reshape((4*N_t, 1))

        # Initialize H_t+1
        H_mat = np.zeros((4*N_t, 3*num_features))

        # Iterate through the observable features
        for n_t in range(0, N_t):

            # Get the Hij_landmark
            Hij_landmark_curr = Hijs_joint[n_t][0]

            # Get the landmark index
            landmark_idx = Hijs_joint[n_t][1]

            # Store the Hij_landmark to H_t+1
            H_mat[4*n_t : 4*(n_t+1), 3*(landmark_idx) : 3*(landmark_idx+1)] = Hij_landmark_curr

        # Get the Kalman Gain, K_t+1|t
        Kalman_Gain = cov @ H_mat.T @ np.linalg.inv(H_mat @ cov @ H_mat.T + np.eye(4 * N_t) * observation_noise_var)

        # Update mew landmarks
        mew_landmarks = mew_landmarks + D_mat @ Kalman_Gain @ (z_tplus1 - tilde_z_tplus1)

        # Update the joint covariance
        cov = (np.eye(3 * num_features) - Kalman_Gain @ H_mat) @ cov

    # Reshape the mean of the landmarks
    mew_landmarks = mew_landmarks.reshape(4, -1, order = 'F')

    return mew_landmarks

def visual_inertial_SLAM(t, linear_velocity, angular_velocity, features, feature_step, K, b, imu_T_cam, motion_noise_var = 1, observation_noise_var = 100):
    '''
    Performing Virtual-Inertial SLAM by performing IMU localization via EKF prediction and Land-mapping via EKF Update Step.
    :Input (Type)   : 1.) t (float)                             - Timestamps
                      2.) linear_velocity (np.ndarray)          - Linear Velocity
                      3.) angular_velocity (np.ndarray)         - Angular Velociry
                      4.) features (np.ndarray)                 - Features from stereo images
                      5.) feature_step (int)                    - Feature step
                      6.) K (np.ndarray)                        - Stereo Camera Intrinsic Matrix, K
                      7.) b (float)                             - Stereo Camera Baseline, b [meters]
                      8.) imu_T_cam (np.ndarray)                - Transformation Matrix from IMU to Camera
                      9.) motion_noise_var (float)              - Motion Noise Variance
                      10.) observation_noise_var (float)        - Sensor noise variance
    :Output(Type)   : 1.) vehicle_poses_imu2world (np.ndarray)  - Vehicle poses in world frame
                      2.) mew_landmarks (np.ndarray)            - Stereo Camera Landmarks (xy-coordinates) in world frame
    '''
    ## Pre-processing Input Data
    #----------------------------
    # Determine the change in time stamps, tau.
    tau = t[:, 1:] - t[:,:-1]
    # Number of interval timestamps
    num_tau = tau.shape[1]
    # Get Smaller Number of Features
    lesser_features = features[:, 1:features.shape[1]:feature_step, :]
    # Number of Features, M
    num_features = lesser_features.shape[1]
    # Discard the first linear and angular velocities
    lin_vel = linear_velocity[:,1:]
    ang_vel= angular_velocity[:,1:]
    # Get the transformation matrix from optical frame to IMU
    cam_T_imu = np.linalg.inv(imu_T_cam)
    # Get the Stereo Camera Calibration Matrix, M
    M_mat = get_stereo_camera_calibration_mat(K, b)
    # Dilation matrix
    D = np.vstack((np.eye(3), np.zeros((1, 3)))) # 4 x 3 
    D_mat = np.tile(D,(num_features, num_features)) # 4M x 3M
    # Define non-observable z
    z_empty = np.array([-1, -1, -1, -1]) 
    # Define projection matrix
    P_mat = np.hstack((np.eye(3), np.zeros((3, 1))))

    ## Initialization Stage
    #----------------------
    # Initialize IMU mean matrix or IMU pose
    mew_IMU = np.eye(4)
    # Initialize Landmark mean matrix or Landmark poses in homogeneous coordinates (4M x 1) instead of (3M x 1)
    mew_landmarks = np.zeros((4 * num_features, 1))
    # Initialize Joint Covariance for both landmark and IMU pose ((3M + 6) x (3M + 6))
    cov = np.eye(3 * num_features + 6)
    # Intialize the landmarks position that check if the feature is initialized.
    mew_init_status = np.zeros(num_features)

    # Initialize vehicle poses
    vehicle_poses_imu2world = np.zeros((4, 4, t.shape[1]))
    vehicle_poses_imu2world[:,:,0] = np.linalg.inv(mew_IMU) 

    ## Start Visual-Inertial SLAM
    for idx in tqdm(range (0, num_tau)):

        ## EKF Prediction Step
        #---------------------
        # Calculate current tau, linear velocity, and angular velocity
        tau_curr = tau[:, idx]
        lin_vel_curr = lin_vel[:, idx]
        ang_vel_curr = ang_vel[:, idx]

        # Generate linear and angular velocities skew-symmetric matrix (v_hat, theta_hat)
        lin_vel_hat_mat = skew_sym_mat3(lin_vel_curr) # 3 x 3
        ang_vel_hat_mat = skew_sym_mat3(ang_vel_curr) # 3 x 3

        # Compute 4 x 4 control hat matrix
        u_hat4 = get_u_hat_mat4(ang_vel_hat_mat, lin_vel_curr)

        # Compute 6 x 6 control curly hat matrix
        u_hat6 = get_u_hat_mat6(ang_vel_hat_mat, lin_vel_hat_mat)

        # Generate random Gaussian motion noise, w_t
        motion_noise_w = np.diag(np.random.normal(0, np.sqrt(motion_noise_var), 6))

        # Calculate the predicted mew and cov matrices of the IMU
        mew_IMU = np.dot(expm(-tau_curr * u_hat4), mew_IMU)
        cov_mat_exp = expm(-tau_curr * u_hat6)
        cov[-6:, -6: ] = np.dot(np.dot(cov_mat_exp, cov[-6:, -6: ]), np.transpose(cov_mat_exp)) + motion_noise_w

        ## EKF Update Step for both IMU and Landmark Mapping
        #----------------------------------------------------
        # Get the IMU pose, T_t (follow the notation from notes) if not, my head is going to explode. too confusing...
        T_t = mew_IMU
        # Extract an array of features at time t, f_t (lazy already, make the notation shorter)
        f_t = lesser_features[:, :, idx]
        # Generate Gaussian observation noise
        observation_noise_v = np.random.normal(0, np.sqrt(observation_noise_var)) # noise

        # Iterate through the landmarks at time t
        for j in range(0, num_features):

            # Initialize Jacobian Observation Matrix, Hijs as a list including the landmark and IMU Hijs with indices. [Hij_landmark, Hij_IMU, j]
            # Note: the Hij's j and j here are different. Don't messed up!
            Hijs_joint = []
            # Initialize observation, z_tplus1 as an empty array
            z_tplus1 = np.array([])
            # Initialize observation with a hat as an empty array
            tilde_z_tplus1 = np.array([])
            
            # If there's are observable features
            if (f_t[:,j] != z_empty).all():

                # If the landmark is not initialized
                if mew_init_status[j] == 0:

                    # Mark as initialize
                    mew_init_status[j] = 1
                    # Initialze the landmark mew
                    mew_landmarks[4*j : 4*(j+1), :] = pixels2world(f_t[:,j], K, b, cam_T_imu, T_t)
                    # Initialize the landmark covariance
                    cov[3*j : 3*(j+1), 3*j : 3*(j+1)] = np.eye(3) * observation_noise_var

                # Update the initialzed mew_landmark and cov_landmark
                else:

                    # Get current landmarks mean
                    curr_mew_landmark = mew_landmarks[4*j : 4*(j+1), :]
                    # Find q_landmark
                    q_landmark = cam_T_imu @ T_t @ curr_mew_landmark
                    # Find pi_q_landmark
                    pi_q_landmark = projection(q_landmark)
                    # Determine the derivative of the projection function dpi(q)/dq
                    dpi_dq_landmark = proj_derivative(q_landmark)
                    # Find z_t+1 
                    z_tp1 = M_mat @ pi_q_landmark + observation_noise_v
                    # Find tilde_z_t+1 
                    tilde_z_tp1 = M_mat @ pi_q_landmark
                    # Store z_t+1
                    z_tplus1 = np.concatenate((z_tplus1, z_tp1), axis=None)
                    # Store tilde_z_t+1
                    tilde_z_tplus1 = np.concatenate((tilde_z_tplus1, tilde_z_tp1), axis=None)
                    # Calculate Hij_landmark
                    Hij_landmark = M_mat @ dpi_dq_landmark @ cam_T_imu @ T_t @ P_mat.T

                    # Find q_pose
                    q_pose = cam_T_imu @  T_t @ curr_mew_landmark
                    # Determine the derivative of the projection function dpi(q)/dq
                    dpi_dq_pose = proj_derivative(q_pose)
                    # Find q_circle
                    q_circle = circle_operator(np.dot(T_t, curr_mew_landmark))
                    # Calculate Hij_pose
                    Hij_pose = - M_mat @ dpi_dq_pose @ cam_T_imu @ T_t @ q_circle

                    # Append to Hijs list
                    Hijs_joint.append((Hij_landmark, Hij_pose, j))

        # Number of observable features, N_t
        N_t = len(Hijs_joint)
        # Reshape z_tplus1 and tilde_z_tplus1
        z_tplus1 = z_tplus1.reshape((4*N_t, 1))
        tilde_z_tplus1 = tilde_z_tplus1.reshape((4*N_t, 1))

        # Initialize H_t+1
        H_mat = np.zeros((4*N_t, 3*num_features+6))

        # Iterate through the observable features
        for n_t in range(0, N_t):

            # Get the Hij_landmark
            Hij_landmark_curr = Hijs_joint[n_t][0]

            # Get the Hij_pose
            Hij_pose_curr = Hijs_joint[n_t][1]

            # Get the landmark index
            landmark_idx = Hijs_joint[n_t][2]

            # Store the Hij_landmark to H_t+1
            H_mat[4*n_t : 4*(n_t+1), 3*(landmark_idx) : 3*(landmark_idx+1)] = Hij_landmark_curr

            # Store the Hij_pose to H_t+1
            H_mat[4*n_t : 4*(n_t+1), -6 : ] = Hij_pose_curr

        # Get the Kalman Gain, K_t+1|t
        Kalman_Gain = cov @ H_mat.T @ np.linalg.inv(H_mat @ cov @ H_mat.T + np.eye(4 * N_t) * observation_noise_var)

        ## Update landmark mew, imu mew, and the joint covariance
        Kdot_1 = Kalman_Gain.dot(z_tplus1 - tilde_z_tplus1)[ : 3*num_features]
        Kdot_2 = Kalman_Gain.dot(z_tplus1 - tilde_z_tplus1)[ -6 : ]

        # Update mew landmarks
        mew_landmarks = mew_landmarks + D_mat @ Kdot_1

        # Update mew IMU
        pos = Kdot_2[ : 3].reshape(3) # Position
        theta = Kdot_2[-3 : ].reshape(3) # Orientation
        exp_term = np.eye(4) + twist(pos, theta) # # Get the exponent term
        mew_IMU = exp_term @ mew_IMU 

        # Update the joint covariance
        cov = (np.eye(3 * num_features + 6) - Kalman_Gain @ H_mat) @ cov

        # Store vehicle pose
        vehicle_poses_imu2world[:, :, idx+1] = np.linalg.inv(mew_IMU) 

    # Reshape the landmark means
    mew_landmarks = mew_landmarks.reshape(4, -1, order = 'F')

    return vehicle_poses_imu2world, mew_landmarks







        





                    
                    

















    


    











    
    
    
    
    
    

    



        

        

        

        
        
