import numpy as np
from numpy.linalg import inv, pinv, svd, norm
from typing import List, Dict
from pr_classes import *
from constants import NUM_LANDMARKS, NUM_POSES
from pr_cast import poses2np, landmarks2np, np2obj


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
        P = np.matmul(P, inv(Xr[:,:,i]))
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
    correspondences = list()
    for i,obs in enumerate(observations):
        if landmark_id in obs.landmark_ids:
            img = obs.img_points[obs.landmark_ids.index(landmark_id)]
            correspondences.append((i, img))
    return correspondences

'''
Compute an estimate of 3D landmark positions from image points.
Input:
-poses: list of robot poses
-observations : list of images
-camera: camera model
-format: indicates how the type of the output should be (either "obj" or "mat")
Output:
-Xl_guess: list of estimated landmarks
'''
def get_landmark_estimates_from_point_cloud(state:Dict[str,list], 
                                            observations:List[Measurement], 
                                            camera:Camera, return_obj:bool=True):
    poses = state["poses"]
    Xr = poses2np(poses)
    Xl_true = landmarks2np(state["landmarks"])
    Xl_guess = np.zeros([3, NUM_LANDMARKS])
    landmark_ids = dict.fromkeys(range(NUM_LANDMARKS), 0) 
    errors = list()

    #compute pseudo-inverse of projection matrix
    P = camera.K @ np.eye(3,4) @ inv(camera.T)
    P = pinv(P)
    
    for l in range(NUM_LANDMARKS):
        point_cloud = []
        correspondences = find_correspondences(l, observations)
        #print("Points in the images associated to landmark {}: {}".format(l,correspondences))

        #if there is no correspondence for a landmark (i.e. it was never seen by the robot), take its actual position
        if len(correspondences) == 0:
            Xl_guess[:,l] = Xl_true[:,l]
        else:
            for i,c in correspondences:
                img_homo = c.get_hom_vec().transpose()
                xr = Xr[:, :, i]
                P = np.matmul(xr,P)
                landmark = np.matmul(P, img_homo)
                point_cloud.append(landmark[:3])
            
            #get the index of the closest point w.r.t. actual position as landmark estimate
            #if the landmark is only seen once, get the only estimate you have,
            #otherwise compute the L2 norm to get the closest point
            if len(point_cloud) == 1:
                Xl_guess[:,l] = point_cloud[0]
            else:
                xl = Xl_true[:,l]
                #print("True landmark: {}".format(xl))
                diff = xl - point_cloud
                #print("diff: {}".format(diff))
                errors = [norm(d, ord=2, axis=0) for d in diff]
                #print(errors)
                m = np.argmin(errors)
                #print(m)
                Xl_guess[:,l] = point_cloud[m]

            #update landmark dictionary
            landmark_ids[l] += 1
    
    if return_obj:
        Xl_guess = np2obj(Xl_guess, 'l')
    return Xl_guess, landmark_ids
