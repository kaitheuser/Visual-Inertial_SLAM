import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt; plt.ion()
import matplotlib
matplotlib.use('Qt5Agg')
from transforms3d.euler import mat2euler

def load_data(file_name):
    '''
    function to read visual features, IMU measurements and calibration parameters
    Input:
        file_name: the input data file. Should look like "XX.npz"
    Output:
        t: time stamp
            with shape 1*t
        features: visual feature point coordinates in stereo images, 
            with shape 4*n*t, where n is number of features
        linear_velocity: velocity measurements in IMU frame
            with shape 3*t
        angular_velocity: angular velocity measurements in IMU frame
            with shape 3*t
        K: (left)camera intrinsic matrix
            with shape 3*3
        b: stereo camera baseline
            with shape 1
        imu_T_cam: extrinsic matrix from (left)camera to imu, in SE(3).
            with shape 4*4
    '''
    with np.load(file_name) as data:
    
        t = data["time_stamps"] # time_stamps
        features = data["features"] # 4 x num_features : pixel coordinates of features
        linear_velocity = data["linear_velocity"] # linear velocity measured in the body frame
        angular_velocity = data["angular_velocity"] # angular velocity measured in the body frame
        K = data["K"] # intrindic calibration matrix
        b = data["b"] # baseline
        imu_T_cam = data["imu_T_cam"] # Transformation from left camera to imu frame 
    
    return t,features,linear_velocity,angular_velocity,K,b,imu_T_cam

def visualize_trajectory_2d(pose, landmarks = None, title_name = "Unknown", path_name = "Unknown", show_ori = False, separate = False, display = True):
    '''
    function to visualize the trajectory in 2D
    Input:
        pose:   4*4*N matrix representing the camera pose, 
                where N is the number of poses, and each
                4*4 matrix is in SE(3)
    '''
    fig,ax = plt.subplots(figsize=(5,5))
    n_pose = pose.shape[2]
    if not separate:
        ax.plot(pose[0,3,:],pose[1,3,:],'r-',label=path_name, zorder=2)
        ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start", zorder=3)
        ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end", zorder=3)

    if landmarks is not None:
        ax.scatter(landmarks[0,:],landmarks[1,:], s=30, marker='*', label="landmark", zorder=1)
  
    if show_ori:
        select_ori_index = list(range(0,n_pose,max(int(n_pose/50), 1)))
        yaw_list = []
        
        for i in select_ori_index:
            _,_,yaw = mat2euler(pose[:3,:3,i])
            yaw_list.append(yaw)
    
        dx = np.cos(yaw_list)
        dy = np.sin(yaw_list)
        dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
        ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
            color="b",units="xy",width=1)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title_name)
    ax.axis('equal')
    ax.grid(False)
    ax.legend()
    plt.savefig(str(title_name)+".png")
    plt.show(block=display)

    return fig, ax

def skew_sym_mat3(vec):
    '''
    Generate 3 x 3 skew-symmetric matrix from a R^3 vector.
    : Input (Type)      - 1.) vec (np.ndarray)           - R^3 Vector
    : Output(Type)      - 1.) hat_mat (np.ndarray)       - 3 x 3 Array
    '''
    # Initialize 3 x 3 skew-symmetric matrix
    hat_mat = np.zeros((3,3))
    
    # Define 3 x 3 skew-symmetric matrix
    hat_mat[0, 1] = -vec[2]
    hat_mat[0, 2] = vec[1]
    hat_mat[1, 0] = vec[2]
    hat_mat[1, 2] = -vec[0]
    hat_mat[2, 0] = -vec[1]
    hat_mat[2, 1] = vec[0]

    return hat_mat

def get_u_hat_mat4(mat1, vec):
    '''
    Generate 4 x 4 generalized velocity from a 3 x 3 skew-symmetric matrix and a R^3 vector.
    : Input (Type)      - 1.) mat1 (np.ndarray)           - 3 x 3 angular velocity skew-symmetric matrix
                          2.) vec (np.ndarray)            - R^3 linear velocity matrix
    : Output(Type)      - 1.) u_hat_mat4 (np.ndarray)     - 4 x 4 Array
    '''
    # Initialize 4 x 4 matrix
    u_hat_mat4 = np.zeros((4,4))

    # Define u_hat4
    u_hat_mat4[:3, :3] = mat1
    u_hat_mat4[:3, 3] = vec

    return u_hat_mat4

def get_u_hat_mat6(mat1, mat2):
    '''
    Generate 6 x 6 generalized velocity from two 3 x 3 skew-symmetric matrices.
    : Input (Type)      - 1.) mat1 (np.ndarray)           - 3 x 3 angular velocity skew-symmetric matrix
                          2.) mat2 (np.ndarray)           - 3 x 3 linear velocity skew-symmetric matrix
    : Output(Type)      - 1.) u_hat_mat (np.ndarray)      - 6 x 6 Array
    '''
    # Initialize 6 x 6 matrix
    u_hat_mat6 = np.zeros((6,6))

    # Define u_hat6
    u_hat_mat6[:3, :3] = mat1
    u_hat_mat6[3:, 3:] = mat1
    u_hat_mat6[:3, 3:] = mat2

    return u_hat_mat6

def get_stereo_camera_calibration_mat(K, b):
    '''
    Generate stereo camera calibration matrix, M
    : Input (Type)      - 1.) K (np.ndarray)                - 3 x 3 Camera Intrinsic Matrix, K
                          2.) b (float)                     - Stereo Camera baseline
    : Output(Type)      - 1.) M_mat (np.ndarray)            - Stereo Camera Calibration Matrix
    '''
    # Initialize a 4 x 4 matrix
    M_mat = np.zeros((4, 4))

    # Define the stereo calibration matrix
    M_mat = np.vstack((K[:2], K[:2]))
    M_mat = np.hstack((M_mat, np.array([0, 0, -K[0, 0] * b, 0]).reshape([4,1])))

    return M_mat

def circle_operator(m):
    '''
    Perform circle operator for m in R^4.
    : Input (Type)      - 1.) m (np.ndarray)                - Homogeneous Coordinate
    : Output(Type)      - 1.) mat (np.ndarray)              - 4 x 6 Array
    '''
    # Extract first three elements (coordinates)
    s = m[:3]

    # Get the skew-symmetric matrix
    s_hat = skew_sym_mat3(s)

    # Extract the last element of the m
    last = m[3]

    # Define the circle operator matrix
    mat = np.hstack([np.eye(3)*last,-s_hat])
    mat = np.vstack([mat,np.zeros([1,6])])

    return mat

def twist(v, w):
    '''
    Map u = [v, w]^T in R^{6x1} to se(3) in 4 x 4
    : Input (Type)      - 1.) u (np.ndarray)                - a vector in R^3
                          2.) v (np.ndarray)                - another vector in R^3
    : Output(Type)      - 1.) twist_mat (np.ndarray)        - 4 x 4 twist matrix 
    '''
    # Initialize the twist matrix
    twist = np.zeros([4,4])

    # Get the skew-symmetric matrix of w and store it in the twist matrix
    twist[:3,:3] = skew_sym_mat3(w)

    # Store the vector v in the twist matrix
    twist[:3,3] = v

    return twist

def projection(q):
    '''
    Get the projection of a vector, q in R4
    : Input (Type)      - 1.) q (np.ndarray)                - a vector in R^4
    : Output(Type)      - 1.) pi_q (np.ndarray)             - a projection vector in R^4
    '''

    # Compute projection of q
    pi_q = q/q[2]

    return pi_q

def proj_derivative(q):
    '''
    Compute the derivative of its projection function q
    : Input (Type)      - 1.) q (np.ndarray)                - a vector in R^4
    : Output(Type)      - 1.) dpi_dq (np.ndarray)           - a 4 x 4 matrix
    '''
    # Initialize the derivative of the projection q.
    dpi_dq = np.zeros([4,4])

    # Compute the derivative of the projection function q
    dpi_dq[0,0] = 1
    dpi_dq[1,1] = 1
    dpi_dq[3,3] = 1
    dpi_dq[0,2] = -q[0]/q[2]
    dpi_dq[1,2] = -q[1]/q[2]
    dpi_dq[3,2] = -q[3]/q[2]
    dpi_dq = dpi_dq / q[2]

    return dpi_dq

def pixels2world(feature, K, b, o_T_i, i_T_w):
    '''
    Convert a pixel coordinate from a stereo camera to the world frame.
    : Input (Type)      - 1.) feature (np.ndarray)          - pixel coordinates from a stereo camera [uL uR vL vR]
                          2.) K (np.ndarray)                - Stereo Camera Intrinsic parameters, K_s
                          3.) b (float)                     - Stereo Camera baseline in meters
                          4.) o_T_i (np.ndarray)            - Transformation Matrix from IMU to Camera frame
                          5.) i_T_w (np.ndarray)            - Transformation Matrix from World to IMU frame
    : Output(Type)      - 1.) dpi_dq (np.ndarray)           - a 4 x 4 matrix
    '''
    # Extract pixel coordinates
    u_L, v_L, u_R, _ = feature

    # Extract the Instrinsic parameters
    fsu = K[0,0]
    fsv = K[1,1]
    cu = K[0,2]
    cv = K[1,2]

    # Get the pixel coordinates in the optical frame
    z = fsu * b / (u_L - u_R)
    y = z * (v_L - cv) / fsv
    x = z * (u_L - cu) / fsu
    
    # Compute homogeneous pixel coordinates in the optical frame
    m_o = np.array([x,y,z,1]).reshape([4,1])

    # Compute the homogeneous pixel coordinates in the world frame
    m_i = np.dot(inv(o_T_i), m_o)
    m_w = np.dot(inv(i_T_w), m_i)

    return m_w











