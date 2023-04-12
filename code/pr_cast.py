import numpy as np
from typing import List
from pr_classes import Landmark, RobotPose, ImagePoint, Camera
from constants import NUM_POSES, NUM_LANDMARKS
from utils import *
    
# #given a landmark expressed in the world frame, returns:
# # -p_img: the 2D projection of the landmark on the image plane
# # -p_cam: the landmark expressed in the camera frame
# def proj2d(camera:Camera, l:Landmark, proj_id:int):
#     #define the world point as a homogeneous point 
#     pw = np.ones(4)
#     pw[:3] = l.get_vec3d()
#     pw = np.transpose(pw)
    
#     #compute the inverse of the transformation to get the transform from world to camera  
#     #and compute the point in the camera world
#     T_inv = np.linalg.inv(camera.T)
#     #print('Inverse transform shape: {}'.format(T_inv.shape))
#     p_cam = camera.K @ np.eye(3,4) @ T_inv @ pw                  # NB: @ same as np.matmul(.)   
#     #print('Image point shape: {}'.format(p_cam.shape))          # expected to be a 4D vector -> drop last coordinate 
#     x_cam, y_cam, z_cam = p_cam[0], p_cam[1], p_cam[2]

#     #project the point in to the 2D image plane
#     x_img = x_cam / z_cam
#     y_img = y_cam / z_cam
#     p_img = ImagePoint(proj_id, x_img, y_img)
#     return p_cam, p_img

# '''
# given an image point, returns the corresponding point in the world (a landmark)
# Input: 
# -img_point: a 2D point
# -lid: landmark ID
# Output:
# -landamark: 3D point representing the landmark
# '''
# def proj3d(camera:Camera, img_point:ImagePoint, lid:int) -> Landmark:
#    return None

#converts a list of RobotPose objects into one 4x4xslices tensor
def poses2np(poses:List[RobotPose], slices:int=NUM_POSES) -> np.array:
    #check dimension match the input list
    assert slices == len(poses)
    out = np.zeros([4,4,slices])
    for i in range(slices):
        pose = poses[i].get_euclidean_param(3)
        out[:, :, i] = v2t(pose)
    return out

#converts a list of Landmark objects into one 3xslices tensor
def landmarks2np(landmarks:List[Landmark], slices:int=NUM_LANDMARKS) -> np.array:
    #check dimension match the input list
    #assert slices == len(landmarks)
    slices = len(landmarks)
    out = np.zeros([3,slices])
    for i in range(slices):
        landmark = landmarks[i].get_vec3d()
        out[:, i] = landmark
    return out

#given a Numpy array of affine robot/landmark positions, returns the list of class objects
def np2obj(input_tensor:np.array, type:str):
    if type not in ['p','l']:
        #print('Unknown class type')
        return []
    elif type == 'l':
        landmarks = list()
        for i in range(input_tensor.shape[1]):
            data = [float(i)]
            data.extend(input_tensor[:, i].reshape(3).tolist())
            landmarks.append(Landmark(data))   
        return landmarks
    elif type == 'p':
        poses = list()
        for i in range(input_tensor.shape[2]):
            data = t2v(input_tensor[:, :, i]).tolist()
            poses.append(RobotPose(data))
        return poses