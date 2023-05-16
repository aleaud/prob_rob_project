import numpy as np
from numpy.linalg import inv, svd, norm
from typing import List, Dict
from pr_classes import *
from constants import NUM_LANDMARKS, NUM_POSES
from pr_cast import *
from errors_jacobians import is_not_visible

'''
Returns estimates of landmark positions in the world 
given robot poses and observations.

Input:
-robot_poses: list of robot poses
-observations: list of measurements from each pose
-cam: camera model

Output:
-lm_guesses: 3xNUM_LANDMARKS matrix of landmark positions found by triangulation
'''
def triangulate(poses:List[RobotPose], observations:List[Measurement], cam:Camera) -> np.array:
    Xr = poses2np(poses)
    Xl_guess = np.zeros([3, NUM_LANDMARKS])
    D = np.zeros([2*NUM_POSES, 4*NUM_LANDMARKS])
    land_index_vec = np.zeros(NUM_LANDMARKS, dtype=int)
    #compute the projection matrix
    P = cam.K @ np.eye(3,4) @ inv(cam.T)                       

    for i in range(len(poses)):
        P = P @ inv(Xr[:,:,i])
        for j in range(len(observations)):
            obs = observations[j]
            for (img,land_id) in zip(obs.img_points, obs.landmark_ids):
                img = img.get_vec()
                if abs(img[0]) > 50 and abs(img[1]) > 50:
                    idx = land_index_vec[land_id]
                    if idx >= 200:
                        break
                    D[2*i, 4*land_id : 4*land_id+4] = img[0] * P[2,:] - P[0,:]
                    D[2*i+1, 4*land_id : 4*land_id+4] = img[1] * P[2,:] - P[1,:]
                    land_index_vec[land_id] += 1
    
    iter = 0
    ids = {}
    for l in range(NUM_LANDMARKS):
        idx = land_index_vec[l]

        #get columns with system to solve and calculate SVD
        A = D[0 : 2*idx, 4*l : 4*l+4]
        _, b, vh = svd(A)

        point = np.array([vh[3,0] / vh[3,3], vh[3,1] / vh[3,3], vh[3,2] / vh[3,3]])
        #landmark_data = [idx, vh[3,0] / vh[3,3], vh[3,1] / vh[3,3], vh[3,2] / vh[3,3]]
        if len(b) == 4:
            Xl_guess[:, iter] = point
            #Xl_guess.append(Landmark(landmark_data))
            ids[l] = iter
            iter += 1
                        
    return ids, Xl_guess[:,0:iter], iter
    
'''
Get the list of all image points referring to a given landmark ID.
Input:
-landmark_id: ID of the target landmark
-observations: list of observations (images), same size of pose list
Output:
-correspondences: list of pose id + image point pairs
'''
def find_correspondences(landmark_id:int, observations:List[Measurement]) -> list:
    correspondences = []
    for i,obs in enumerate(observations):
        if landmark_id in obs.landmark_ids:
            img = obs.img_points[obs.landmark_ids.index(landmark_id)]
            correspondences.append((i, img))
    return correspondences

'''
Compute an estimate of 3D landmark positions from all image points.
Input:
-poses: list of robot poses
-observations : list of images
-camera: camera model
-return_obj: boolean flag which indicates what the type of the output should be 
    if True returns list of landmarks, otherwise return stack of 3 dimensional column vectors

Output:
-Xl_guess: list of estimated landmarks
-landmark_ids: dictionary in which we store the amount of observed point for each seen landmark
'''
def triangulate_pc(state:Dict[str,list], observations:List[Measurement], camera:Camera, return_obj:bool=True):

    poses = state["poses"]
    Xr = poses2np(poses)
    Xl_true = landmarks2np(state["landmarks"])
    Xl_guess = np.zeros([3, NUM_LANDMARKS])
    landmark_system = np.zeros([2*NUM_POSES, 4*NUM_LANDMARKS])          #why? Robot poses are 2D signals, landmarks expressed in homogeneous coordinates
    landmark_ids = dict.fromkeys(range(NUM_LANDMARKS), 0) 
    img_point_idx = 0

    #compute projection matrix only once!
    proj_mat = camera.K @ np.eye(3,4) @ inv(camera.T)

    for l in range(NUM_LANDMARKS):
        xl = np.append(Xl_true[:,l], 1)
        correspondences = find_correspondences(l, observations)
        #print("Points in the images associated to landmark {}: {}".format(l,correspondences))

        #check whether there is at least one correspondence for a landmark (i.e. it was never seen by the robot)
        if len(correspondences) > 0:
            for pose_idx, obs_img_point in correspondences:
                obs_img_point = obs_img_point.get_vec()
                xr = Xr[:, :, pose_idx]
                #move the projection matrix to the current noisy robot pose
                P = proj_mat @ inv(xr)

                #estimate landmark point on the image plane
                landmark_point = P @ xl
                landmark_point = landmark_point[:2] / landmark_point[2]
                img_point = ImagePoint(img_point_idx, landmark_point[0], landmark_point[1])

                #filter out the bad estimates e.g. take only the visible points in the image plane
                if not is_not_visible(camera, img_point):
                    #print('Estimate {} at pose {} for landmark {}: {}'.format(img_point_idx, pose_idx, l, img_point.get_vec()))
                    row = 2*pose_idx
                    col = 4*l
                    landmark_system[row,   col:col+4] = img_point.w * P[2,:] - P[0,:]
                    landmark_system[row+1, col:col+4] = img_point.h * P[2,:] - P[1,:]
                    img_point_idx += 1
                    landmark_ids[l] += 1

    counter = 0
    seen_landmarks = {}
    for l in range(NUM_LANDMARKS):
        row_idx = landmark_ids[l]
        _, b, vh = svd(landmark_system[0:2*row_idx, 4*l : 4*l+4])
        
        #check whether the estimated position is valid
        x = vh[3,0] / vh[3,3]
        y = vh[3,1] / vh[3,3]
        z = vh[3,2] / vh[3,3]
        est_landmark = np.array([x,y,z])
        if len(b)==4:
            Xl_guess[:, counter] = est_landmark
            seen_landmarks[l] = counter
            counter += 1

    #cast output if required
    if return_obj:
        Xl_guess = np2obj(Xl_guess, 'l')
    return Xl_guess, seen_landmarks
