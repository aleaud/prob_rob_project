#from operator import gt
import math
import numpy as np
from typing import List



''' CLASSES FOR ROBOT ENVIRONMENT '''
class Landmark():
    # landmarks are 3D points expressed in the world frame
    def __init__(self, input_line:list) -> None:
        self.id = int(input_line[0])
        self.x = float(input_line[1])
        self.y = float(input_line[2])
        self.z = float(input_line[3])
    
    def get_id(self):
        return self.id
    
    def get_vec3d(self):
        return np.array([self.x, self.y, self.z])

    def get_vec6d(self):
        return np.array([self.x, self.y, self.z, 0, 0, 0])


''' CLASS FOR EVALUATION '''
class World():
    def __init__(self) -> None:
        self.landmarks = []

    #fill the world with landmarks whose data comes from a file
    def setup(self, input_file:str) -> None:
        with open(input_file) as f:
            self.landmarks = [Landmark(line.strip().split()) for line in f.readlines()]
            #f.close()
    
    #add a new landmark in the list of landmarks
    #if p is specified, add the new landmark at position p, otherwise add at the end
    def add(self, landmark:Landmark, p:int=-1):
        if p == -1:
            self.landmarks.append(landmark)
        elif p in range(len(self.landmarks)):
            self.landmarks.insert(p, landmark)
    
    #remove a landmark at position p
    def remove_by_pos(self, p:int):
        assert p >= 0 and p < len(self.landmarks)
        self.landmarks.pop(p)
    
    #remove the landmark with specific id
    def remove_by_id(self, id:int):
        for i in range(len(self.landmarks)):
            curr_id = self.landmarks[i].get_id()
            if curr_id == id:
                self.landmarks.pop(i)
                break
        print(f'No landamark with ID {id} is in the world...')

''' CLASS FOR SENSOR MODEL '''
# define the (fixed) pinhole camera model from data gathered by camera.dat
class Camera():

    # camera matrix
    K = np.array([[180, 0,  320], 
                  [ 0, 180, 240], 
                  [ 0,  0,   1 ]], dtype=np.intc)           
    
    # pose of the camera w.r.t. robot
    T = np.array([[ 0,  0,  1, 0.2],
                  [-1,  0,  0,  0 ],
                  [ 0, -1,  0,  0 ],
                  [ 0,  0,  0,  1 ]], dtype=np.intc)
    
    # how close/far the camera can perceive stuff
    z_near = 0
    z_far = 5

    # image size (width=cols, height=rows)
    width = 640
    height = 480


#nothing but an unique 2D point
class ImagePoint():
    def __init__(self, id:int, w:float, h:float) -> None:
        self.id = id
        self.w = w
        self.h = h
    
    #returns data in NumPy 2D array format
    def get_vec(self) -> np.array:
        return np.array([self.w, self.h])

    #returns data in homogeneous coordinates
    def get_hom_vec(self) -> np.array:
        return np.array([self.w, self.h, 1])

''' MEASUREMENT CLASS
It stores the following attributes:
-timestep: observations at time t
-association_vector: a[j] = i represents that i-th image point refers to j-th landmark
-img_points: list of image points 
'''
class Measurement():
    def __init__(self, input_file:str) -> None:
        self.img_points = []
        self.landmark_ids = []
        self._read_data(input_file)

    def _read_data(self, input_file:str) -> None:
        with open(input_file, 'r') as mf:
            lines = mf.readlines()
            for line in lines:
                line = line.strip().split()
                if len(line) > 0:
                    #ignore poses
                    if line[0] == 'seq:':
                        # we can retrieve robot pose from timestep!
                        self.timestep = int(line[1])
                    elif line[0] == 'point': 
                        #ith element (image point ID) refers to actual landmark ID -> data association
                        img_point_id = int(line[1])
                        landmark_id = int(line[2])
                        
                        #image point coordinates on image plane
                        w = float(line[3])
                        h = float(line[4])
                        img_point = ImagePoint(img_point_id,w,h)
                        
                        #get associations from here!
                        self.img_points.append(img_point)
                        self.landmark_ids.append(landmark_id)


''' CLASS FOR MANIFOLD SE(2) OBJECT '''
class RobotPose():
    def __init__(self, data:List[float]) -> None:
        # translational part
        self.x, self.y = float(data[0]), float(data[1])
        self.t = np.array([self.x, self.y])
        
        # rotational part
        self.theta = float(data[2])
        s,c = math.sin(self.theta), math.cos(self.theta)
        self.R = np.array([[c, -s],[s, c]])
    
    #returns the 3d vector representing its Euclidean parametrization 
    # easier to use 3D parametrization to be aligned with the state space
    def get_euclidean_param(self, dim:int=3) -> np.array:
        if dim==2:
            return np.array([self.x, self.y, self.theta])
        elif dim==3:
            #neither it moves along z axis, nor rotates around x and y axes (similar to t2v)
            return np.array([self.x, self.y, 0, 0, 0, self.theta])
        else:
            print(f'Robot vector dimension {dim} is not supported for this task. Empty 6D vector is returned.')
            return np.zeros(6)
        

''' ROBOT TRAJECTORY CLASS '''        
# store robot poses coming from noisy odometry and ground truth poses (for evaluation only)
class RobotTrajectory():
    def __init__(self, input_file:str) -> None:
        self._read_data(input_file)

    #initialize data
    def _read_data(self, input_file:str) -> None:
        self.robot_poses = []
        with open(input_file) as rtf:
            for line in rtf.readlines():
                line = line.strip().split()
                id = int(line[0])    
                odom_robot_pose = RobotPose(line[1:4])              # store position estimated from noisy odometry
                gt_robot_pose = RobotPose(line[4:])                 # store ground truth position
                self.robot_poses.append({"id" : id, 
                                         "odom" : odom_robot_pose,  #.get_euclidean_param(3), 
                                         "gt" : gt_robot_pose})     #.get_euclidean_param(3)})    
        

    # get dictionary corresponding to the robot pose info at time t in the trajectory
    def get_pose(self, t:int) -> dict:
        return self.robot_poses[t]              #NB: position in the list is equal to the id field!
    
    # compute the error distance between odometry and ground truth measures at time t
    # to adjust, need to include the least squares solution to be evaluated with the ground truth!
    # use Ctrl+K+C to comment, and Ctrl+K+U to uncomment this section ;)
    def distance_error(self, t:int, to_add:bool=False) -> float:
        assert t > 0 and t <= len(self.robot_poses)
        pose = self.get_pose(t)
        #assert pose.get('id') == t
        odom = pose.get('odom')
        gt = pose.get('gt')
        diff = odom - gt
        error = np.linalg.norm(diff, ord=2, axis=0)
        if to_add:
            self.robot_poses[t].update({'error' : error})
        return error


            