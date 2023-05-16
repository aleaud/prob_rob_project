import os
from tls import tls
from pr_plot import *
from estimates import triangulate
from pr_classes import *
from utils import *
from constants import *
from pr_cast import *

#define the world
env = World()
env_input_file = os.path.join(DATASET_PATH, WORLD_FILE)
env.setup(env_input_file)

#get noisy robot poses from odometry from the trajectory file (for evaluation only)
traj_input_file = os.path.join(DATASET_PATH, TRAJ_FILE)
trajectory = RobotTrajectory(traj_input_file)


#define the camera model
camera = Camera()

#domains definition
#state space
X = {"poses" : [trajectory.get_pose(i).get('odom') for i in range(len(trajectory.robot_poses))],
     "landmarks" : env.landmarks}

#measurement space 
Z = []
for i in range(NUM_POSES):
    idx = get_meas_file_idx(i)
    file = MEAS_FILE.format(idx)
    meas_input_file = os.path.join(DATASET_PATH, file)
    meas = Measurement(meas_input_file)
    Z.append(meas)

#test measurements
# print(f'These are the first {M} measurements:')           
# for i in range(M):
#     test_meas = Z[i]
#     print(f'Measurements at time {test_meas.timestep}')
#     for m in range(M):
#         ip = test_meas.img_points[m]
#         print('\t-coordinates of image point {}: width={}, height={}'.format(m, ip.w, ip.h))

#triangulate landmark positions to get an initial estimate 
seen_landmark_ids, Xl_est, nl = triangulate(X["poses"], Z, camera)
print("Landmark 2D positions have been estimated via triangulation.")
Xl_est = np2obj(Xl_est, 'l')
#save gold landmarks and use the estimated ones to perform TLS
Xl_true = X["landmarks"]
X["landmarks"] = Xl_est

#get projection association vector
#count total number of landmark measurements
tot_meas = [len(meas.landmark_ids) for meas in Z]
tot_meas = np.asarray(tot_meas).sum()
Z_proj = np.zeros([PROJ_DIM, tot_meas])
projection_associations = np.zeros([PROJ_DIM, tot_meas]).astype(int)
n_meas = 0
for i in range(len(Z)):
    z = Z[i]
    obs_lm = z.landmark_ids
    obs_img = z.img_points
    for j in range(len(obs_lm)):
        if obs_lm[j] in seen_landmark_ids.keys():
            lid = seen_landmark_ids[obs_lm[j]]
            projection_associations[:, i] = [i, lid]
            if lid in obs_lm:
                Z_proj[:, n_meas] = obs_img[obs_lm.index(lid)].get_vec()
                n_meas += 1
projection_associations = projection_associations[:, 0:n_meas]
Z_proj = Z_proj[:, 0:n_meas]


#compute robot-robot relative positions and get pose association vector
n = NUM_POSES-1
Zr = np.zeros([4,4,n])
pose_associations = np.zeros([PROJ_DIM, n]).astype(int)
poses = poses2np(X["poses"])
for i in range(n):
    Xi = poses[:, :, i]
    Xj = poses[:, :, i+1]
    pose_associations[:, i] = [i, i+1]
    #compute relative position of next pose from previous one
    Zr[:, :, i] = np.linalg.inv(Xi) @ Xj

#set parameters
damping_factor = 1e-2
kernel_threshold_proj = 5000
kernel_threshold_pose = 0.01
max_iters = 5
#error_threshold = 1e-4

Xr, Xl, chi_stats_proj, inliers_proj, chi_stats_poses, inliers_poses, H, b, it =\
    tls(X, Z, Zr, projection_associations, pose_associations, camera, max_iters, damping_factor, 
        #error_threshold, 
        kernel_threshold_proj, kernel_threshold_pose)


#plot section
plot_landmarks(Xl_true, Xl_est, Xl)
plot_trajectories(trajectory, Xr)
#plot_chi_inliers(chi_stats_poses, chi_stats_proj, inliers_poses, inliers_proj, it)