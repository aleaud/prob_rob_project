import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union
from mpl_toolkits.mplot3d import Axes3D
from pr_classes import RobotPose, Landmark, RobotTrajectory
from pr_cast import landmarks2np, poses2np
from constants import NUM_POSES
from utils import t2v

#paths to save plots
TRIANG_PATHS = ["../imgs/landmarks_triang.png",
                "../imgs/trajectories_triang.png",
                "../imgs/chi_inliers_triang.png"]

PC_PATHS = ["../imgs/landmarks_pc.png",
            "../imgs/trajectories_pc.png",
            "../imgs/chi_inliers_pc.png"]

#function to plot landmarks resulting from TLS, estimated landmarks and ground truth landmarks
def plot_landmarks(true:Union[List[Landmark], np.array], 
                   est:Union[List[Landmark], np.array], 
                   pred:Union[List[Landmark], np.array],
                   what:str='tr') -> None:
    
    #force to being matrix
    if not isinstance(true, (np.ndarray, np.generic)):
        true = landmarks2np(true, 'l')
    
    if not isinstance(est, (np.ndarray, np.generic)):
        est = landmarks2np(est, 'l')
    
    if not isinstance(pred, (np.ndarray, np.generic)):
        pred = landmarks2np(pred, 'l')    

    fig = plt.figure(1)
    fig.set_size_inches(16, 12)
    fig.suptitle("Landmarks", fontsize=16)

    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(true[0,:], true[1,:], true[2,:], 'o', mfc='none', color='b', markersize=3)
    ax1.plot(est[0, :], est[1, :], est[2, :], 'x', color='r', markersize=3)
    ax1.set_title("Landmark ground truth (blue) and triangulation (red)", fontsize=10)

    ax2 = fig.add_subplot(222, projection='3d')
    ax2.plot(true[0,:], true[1,:], true[2,:], 'o', mfc='none', color='b', markersize=3)
    ax2.plot(pred[0,:], pred[1,:], pred[2,:], 'x', color='r', markersize=3)
    ax2.set_title("Landmark ground truth (blue) and optimization (red)", fontsize=10)

    ax3 = fig.add_subplot(121)
    ax3.plot(true[0,:],true[1,:], 'o', mfc='none', color='b', markersize=3)
    ax3.plot(est[0, :], est[1, :], 'x', color='r', markersize=3)
    ax3.set_title("Landmark ground truth (blue) and triangulation (red)", fontsize=10)
    ax3.axis([-15,15,-15,15])

    ax4 = fig.add_subplot(122)
    ax4.plot(true[0,:],true[1,:], 'o', mfc='none', color='b', markersize=3)
    ax4.plot(pred[0,:], pred[1,:], 'x', color='r', markersize=3)
    ax4.axis([-15,15,-15,15])
    ax4.set_title("Landmark ground truth (blue) and optimization (red)", fontsize=10)

    path = TRIANG_PATHS[0] if what == 'tr' else PC_PATHS[0]
    fig.savefig(path, dpi=100)
    plt.show()

#function to plot robot poses
def plot_trajectories(traj:RobotTrajectory, pred:Union[List[RobotPose], np.array], what:str='tr'):
    
    traj_true_mat = [traj.robot_poses[i].get('gt') for i in range(len(traj.robot_poses))]
    traj_true_mat = poses2np(traj_true_mat)
    traj_guess_mat = [traj.robot_poses[i].get('odom') for i in range(len(traj.robot_poses))]
    traj_guess_mat = poses2np(traj_guess_mat)
    traj_est_mat = poses2np(pred) if not isinstance(pred, (np.ndarray, np.generic)) else pred
    
    traj_true = np.zeros([3,NUM_POSES])
    traj_guess = np.zeros([3,NUM_POSES])
    traj_est = np.zeros([3,NUM_POSES])
    for i in range(NUM_POSES):
        traj_true[:,i] = t2v(traj_true_mat[:,:,i])[0:3]
        traj_guess[:,i] = t2v(traj_guess_mat[:,:,i])[0:3]
        traj_est[:,i] = t2v(traj_est_mat[:,:,i])[0:3]

    fig = plt.figure(1)
    ax1 = fig.add_subplot(223)
    ax1.plot(traj_true[0,:],traj_true[1,:], 'o', mfc='none', color='b', markersize=3)
    ax1.plot(traj_guess[0,:],traj_guess[1,:], 'x', color='r', markersize=3)
    ax1.axis([-10,10,-10,10])
    ax1.set_title("Robot ground truth (blue) and odometry values (red)", fontsize=10)

    ax2 = fig.add_subplot(224)
    ax2.plot(traj_true[0,:],traj_true[1,:], 'o', mfc='none', color='b', markersize=3)
    ax2.plot(traj_est[0,:],traj_est[1,:], 'x', color='r', markersize=3)
    ax2.axis([-10,10,-10,10])
    ax2.set_title("Robot ground truth (blue) and optimization (red)", fontsize=10)

    path = TRIANG_PATHS[1] if what == 'tr' else PC_PATHS[1]
    fig.savefig(path, dpi=100)
    plt.show()

#function to plot chi and inliers
def plot_chi_inliers(chi_poses:np.array, chi_lm:np.array, inliers_poses:np.array, inliers_lm:np.array, iter:int, what:str='tr'):

    fig = plt.figure(2)
    fig.set_size_inches(16, 12)
    fig.suptitle("Chi and Inliers", fontsize=16)

    ax1 = fig.add_subplot(221)
    ax1.plot(chi_poses[0:iter])
    ax1.set_title("Chi poses", fontsize=10)
    ax2 = fig.add_subplot(222)
    print(inliers_poses)
    ax2.plot(inliers_poses[0:iter])
    ax2.set_title("Inliers poses", fontsize=10)

    ax3 = fig.add_subplot(223)
    ax3.plot(chi_lm[0:iter])
    ax3.set_title("Chi projections", fontsize=10)
    ax4 = fig.add_subplot(224)
    ax4.plot(inliers_lm[0:iter])
    ax4.set_title("Inliers projections", fontsize=10)

    path = TRIANG_PATHS[2] if what == 'tr' else PC_PATHS[2]
    fig.savefig(path, dpi=100)
    plt.show()

