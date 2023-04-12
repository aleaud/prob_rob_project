import math
import numpy as np
from typing import List, Dict
from pr_classes import *
from utils import *
from errors_jacobians import *
from constants import *
from pr_cast import poses2np, landmarks2np, np2obj


''' Defines the boxplus operator which applies a perturbation to a set of poses and landmarks 
Input:
-Xr: list of robot poses
-Xl: list of landmark poses
-dx: state (poses + landmarks) perturbation vector
-return_np: return NumPy arrays if True, list of class objects otherwise

Output:
-perturbed_poses: 4x4xNUM_POSES tensor of robot poses in affine form after perturbation
-perturbed_landmarks: 3xNUM_LANDMARKS of landmark poses after perturbation
'''
def boxplus(Xr:List[RobotPose], 
            Xl:List[Landmark], 
            dx:np.array,
            pert_limit:float=1e-3, 
            return_np:bool=True):
    
    perturbed_poses = poses2np(Xr, len(Xr))
    perturbed_landmarks = landmarks2np(Xl, len(Xl))

    for i in range(len(Xr)):
        pose_idx = get_pert_pose_idx(i)
        dxr = dx[pose_idx : pose_idx + POSE_DIM]
        #print("Pose perturbation {}: {}".format(i, dxr))
        #curr_pose = perturbed_poses[:, :, i]
        p = np.linalg.norm(dxr)
        if p > pert_limit:
          perturbed_poses[:, :, i] = np.matmul(v2t(dxr), perturbed_poses[:, :, i])
    
    for i in range(len(Xl)):
        lm_idx = get_pert_landmark_idx(i)
        dxl = dx[lm_idx : lm_idx + LANDMARK_DIM]
        #print("Landmark perturbation {}: {}".format(i, dxl))
        p = np.linalg.norm(dxl)
        if p > pert_limit:
          perturbed_landmarks[:,i] += dxl.reshape(3)

    if return_np:
      return perturbed_poses, perturbed_landmarks
    return np2obj(perturbed_poses, type='p'), np2obj(perturbed_landmarks, type='l')

'''  
Perform linearization step of TLS algorithm on robot-landmark measurements.
This function represents one iteration of LS from the landmark perspective.
Input:
-state: dictionary containing robot poses and landmarks
-observations: list of measurements
-associations: s.t. associations(:,k)=[p_idx,l_idx]' means the kth measurement
              refers to an observation made from pose p_idx, that observed landmark l_idx
-kernel_threshold: for robustifier

Output:
-H: the H matrix, filled
-b: the b vector, filled
-chi_tot: the total chi2 of the current round
-num_inliers: number of measurements whose error is below kernel_threshold
'''
def linearize_projections(state:Dict[str, list], 
                          observations:List[Measurement],
                          associations:np.array,
                          cam:Camera,
                          kernel_threshold:int=5000):
    
  #sanity check
  r = state.get('poses')          #list of robot poses
  l = state.get('landmarks')      #list of landmarks

  #clear H and b
  H = np.zeros([SYSTEM_SIZE, SYSTEM_SIZE])
  b = np.zeros([SYSTEM_SIZE, 1])
  
  #to count inliers
  chi_tot = 0
  num_inliers = 0

  for i in range(len(observations)):
      obs = observations[i]
      for j in range(len(obs.img_points)):
        z = obs.img_points[j]
        pose_id = associations[0, i]
        landmark_id = associations[1, i]
        
        is_valid, e, Jr, Jl = projection_error_and_jacobian(r[pose_id], l[landmark_id], z, cam)
        if is_valid:
          chi = np.matmul(np.transpose(e), e)
          if chi > kernel_threshold:
              e *= math.sqrt(kernel_threshold / chi)
              chi = kernel_threshold
          else:
              num_inliers += 1
          chi_tot += chi
      
          #define omega matrix: noise on odometry is 0.001
          O = 0.0001 * np.identity(2)

          #update H and b at corresponding positions
          pose_idx = get_pert_pose_idx(pose_id)
          landmark_idx = get_pert_landmark_idx(landmark_id)

          #update H w.r.t. robot pose
          H[pose_idx     : pose_idx + POSE_DIM, 
            pose_idx     : pose_idx + POSE_DIM] += Jr.transpose() @ O @ Jr
          H[pose_idx     : pose_idx + POSE_DIM, 
            landmark_idx : landmark_idx + LANDMARK_DIM] += Jr.transpose() @ O @ Jl
          
          #update H w.r.t. landmark pose
          H[landmark_idx : landmark_idx + LANDMARK_DIM, 
            landmark_idx : landmark_idx + LANDMARK_DIM] += Jl.transpose() @ O @ Jl
          H[landmark_idx : landmark_idx + LANDMARK_DIM, 
            pose_idx     : pose_idx + POSE_DIM] += Jl.transpose() @ O @ Jr            

          b[pose_idx     : pose_idx + POSE_DIM] += Jr.transpose() @ O @ e
          b[landmark_idx : landmark_idx + LANDMARK_DIM] += Jl.transpose() @ O @ e
  
  #for outliers: do nothing
  return H, b, chi_tot, num_inliers

'''  
Performs linearization step of TLS algorithm on robot-robot measurements.
This function represents one iteration of LS from the poses point of view.
Input:
-state: dictionary containing robot poses and landmark estimates
-rel_poses: matrix of relative positions between subsequent robot poses
-observations: list of measurements
-associations: association vector s.t. associations(:,k)=[i, j]' 
  means the kth measurement refers to an observation made from pose i that observes pose j
-kernel_threshold: for robustifier

Output:
-H: the H matrix, filled
-b: the b vector, filled
-chi_tot: the total chi2 of the current round
-num_inliers: number of measurements whose error is below kernel_threshold
'''
def linearize_poses(state:Dict[str, list], 
                    rel_poses:np.array,
                    associations:np.array,
                    kernel_threshold:float=0.01):
    
    # clear H and b
    H = np.zeros([SYSTEM_SIZE, SYSTEM_SIZE])
    b = np.zeros([SYSTEM_SIZE, 1])
    chi_tot = 0
    num_inliers = 0
    
    poses = state.get('poses')
    for k in range(len(poses)-1):
      #initialize omega matrix
      O = np.eye(12)
      
      #get index of robot pose i (observer) and j (observed)
      pi = associations[0, k]
      pj = associations[1, k]
      z  = rel_poses[:,:,k]
      Xi = poses[pi]
      Xj = poses[pj]
      e, Ji, Jj = pose_error_and_jacobian(Xi, Xj, z)

      # compute chi and count inliers
      chi = np.transpose(e) @ O @ e
      is_inlier = True
      if chi > kernel_threshold:
        #update Omega matrix
        O = O * math.sqrt(kernel_threshold/chi)
        chi = kernel_threshold
        is_inlier = False
      else:
        num_inliers += 1
        chi_tot += chi
      if not is_inlier:
        continue

      #get indices from perturbation vector
      ith_pose_idx = get_pert_pose_idx(pi)
      jth_pose_idx = get_pert_pose_idx(pj)

      #fill H and b
      H[ith_pose_idx : ith_pose_idx + POSE_DIM,
        ith_pose_idx : ith_pose_idx + POSE_DIM] += Ji.transpose() @ O @ Ji

      H[ith_pose_idx : ith_pose_idx + POSE_DIM,
        jth_pose_idx : jth_pose_idx + POSE_DIM] += Ji.transpose() @ O @ Jj

      H[jth_pose_idx : jth_pose_idx + POSE_DIM,
        ith_pose_idx : ith_pose_idx + POSE_DIM] += Jj.transpose() @ O @ Ji

      H[jth_pose_idx : jth_pose_idx + POSE_DIM,
        jth_pose_idx : jth_pose_idx + POSE_DIM] += Jj.transpose() @ O @ Jj

      b[ith_pose_idx : ith_pose_idx + POSE_DIM] += Ji.transpose() @ O @ e
      b[jth_pose_idx : jth_pose_idx + POSE_DIM] += Jj.transpose() @ O @ e

    return H, b, chi_tot, num_inliers

'''
Implementation of the optimization loop with robust kernel
applies a perturbation to a set of landmarks and robot poses
Input:
-state: initial robot poses and landmark estimates
-observations: list of measurements
-relative_positions: tensor of relative positions between consecutive robot poses
-projection/pose associations: association matrices
-iterations: maximum number of iterations performing the algorithm
-dmp: damping factor (in case system not spd)
-kernel_threshod_proj: to robustify projections
-kernel_threshod_pos: to robustify poses

Output:
-Xr: robot poses after optimization
-Xl: landmarks after optimization
-chi_stats_{l,p,r}: array 1:num_iterations, containing evolution of chi2 for landmarks, projections and poses
-num_inliers{l,p,r}: array 1:num_iterations, containing evolution of inliers landmarks, projections and poses
-iteration: last iteration of TLS
-H,b: output of the TLS
'''
def tls(state:Dict[str, list], 
        observations:List[Measurement],
        relative_positions:np.array,
        proj_associations:np.array,
        poses_associations:np.array,
        cam: Camera,
        iterations:int=1,
        dmp:float=0.0,
        error_threshold:float=1e-4,
        kernel_threshold_proj:int=0,
        kernel_threshold_pos:float=0.0):
    
    #initialize chi and inlier containers
    chi_stats_proj = np.zeros(iterations)
    num_inliers_proj = np.zeros(iterations)
    chi_stats_poses = np.zeros(iterations)
    num_inliers_poses = np.zeros(iterations)

    it=0
    error = 1e6
    print("Start Total Least Squares...")
    while it < iterations and error > error_threshold:
      #get linearized projections and update chi and inliers info
      H_proj, b_proj, chi_proj, inliers_proj = linearize_projections(state, observations, 
                                                                    proj_associations, cam,
                                                                    kernel_threshold_proj)
      chi_stats_proj[it] += chi_proj
      num_inliers_proj[it] += inliers_proj

      #get linearized poses and update chi and inliers info
      H_poses, b_poses, chi_poses, inliers_poses = linearize_poses(state,
                                                                  relative_positions, 
                                                                  poses_associations,
                                                                  kernel_threshold_pos)
      chi_stats_poses[it] += chi_poses
      num_inliers_poses[it] += inliers_poses

      #build H and b
      H = H_poses + H_proj
      H += np.eye(SYSTEM_SIZE)*dmp
      b = b_poses + b_proj

      #solve linear system -> compute optimal perturbation
      dx = np.zeros([SYSTEM_SIZE, 1])
      dx[POSE_DIM:] = -np.linalg.solve(H[POSE_DIM:, POSE_DIM:], b[POSE_DIM:, 0]).reshape([-1,1])

      #apply perturbation
      Xr, Xl = boxplus(state["poses"], state["landmarks"], dx, True)
      state["poses"] = np2obj(Xr, 'p')
      state["landmarks"] = np2obj(Xl,'l')
      
      error = np.sum(np.absolute(dx))
      print(f"Iteration {it} -> error: {error}")
      it+=1
    
    #resume
    it_left = iterations - it
    print("Total Least Squares stops after {} iterations ({} are left) with a final error {} m".format(it, it_left, error))
    return Xr, Xl, chi_stats_proj, inliers_proj, chi_stats_poses, inliers_poses, H, b, it

        
