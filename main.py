from pr3_utils import *
from Visual_Inertial_SLAM import *


"""
Debugging Settings
"""
data_describe = False

"""
Virtual-Inertial SLAM Settings
"""
feature_step = 10			# Feature step
display_plots = True		# Display plots setting (True = Yes, False = No)


if __name__ == '__main__':

	# Filenames
	filenames = ["03.npz", "10.npz"]

	for filename in filenames:

		# Load the measurements
		print("\n" +"*" * 70)
		print("Loading file " + filename + "...")
		print("*" * 70)
		file_dir = "./data/" + filename
		t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(file_dir)
		print("Data Loading Completed...")
		print("*" * 70 + "\n")

		# Print how the load data look like
		if data_describe:
			print("*" * 70)
			print("Data Description:")
			print("*" * 70)
			# Time stamps in UNIX standard seconds-since-the-epoch January 1, 1970.
			print("Time Stamps Data Shape: " + str(t.shape))
			# Visual features across different time stamps
			print("Features Data Shape: " + str(features.shape))
			# Linear Velocty in the body frame of the IMU
			print("Linear Velocity Data Shape: " + str(linear_velocity.shape))
			# Angular Velocity in the body frame of the IMU
			print("Angular Velocity Data Shape: " + str(angular_velocity.shape))
			# Stereo Camera Instrinsic Matrix, K
			print("Instrinsic Calibration: " + str(K))
			# Stereo Camera Baseline in meters
			print("Baseline: " + str(b) + " meters")
			# Transformation Matrix from Camera to IMU
			print("Transformation Matrix from Camera to IMU: " + str(imu_T_cam))
			print("*" * 70)

		# (a) IMU Localization via EKF Prediction
		vehicle_poses_imu2world, vehicle_poses_world2imu = IMU_localization_EKF_predict(t, linear_velocity, angular_velocity, motion_noise_var = 1)
		visualize_trajectory_2d(vehicle_poses_imu2world, landmarks = None, title_name="IMU Localization via EKF Prediction (Data " + filename + ")",
								path_name = "Estimated Trajectory", show_ori = True, separate = False, display = display_plots)
		print("Completed processing IMU Localization via EKF Prediction.\n")

		# (b) Landmark Mapping via EKF Update
		landmark_poses = visual_mapping_EKF_update(features, feature_step, vehicle_poses_world2imu, K, b, imu_T_cam, observation_noise_var = 100)
		visualize_trajectory_2d(vehicle_poses_imu2world, landmarks = landmark_poses, title_name="Landmark Mapping via EKF Update (Data " + filename + ")",
								path_name = "Estimated Trajectory", show_ori = False, separate = True, display = display_plots)
		print("Completed Landmark Mapping via EKF Update.\n")

		# (c) Visual-Inertial SLAM
		vehicle_poses_VSLAM, landmark_poses_VSLAM = visual_inertial_SLAM(t, linear_velocity, angular_velocity, features, feature_step, K, b, imu_T_cam, motion_noise_var = 1, observation_noise_var = 100)
		visualize_trajectory_2d(vehicle_poses_VSLAM, landmarks = landmark_poses_VSLAM, title_name="Visual-Inertial SLAM (Data " + filename + ")",
								path_name = "Estimated Trajectory", show_ori = True, separate = False, display = display_plots)
		print("Completed Visual-Inertial SLAM.\n")
	


