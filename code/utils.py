# here we define some auxiliary methods and constants for the TLS formulation
import math
import numpy as np
from constants import NUM_LANDMARKS, NUM_POSES, POSE_DIM, LANDMARK_DIM


''' 3D ROTATION MATRICES '''
def Rx(theta:float):
    c = math.cos(theta)
    s = math.sin(theta)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return R

# Rotation matrix around y axis
def Ry(theta:float):
    c = math.cos(theta)
    s = math.sin(theta)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return R

# Rotation matrix around z axis
def Rz(theta:float):
    c = math.cos(theta)
    s = math.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return R

''' PRECOMPUTED DERIVATIVES OF 3D ROTATION MATRICES AT THE ORIGIN '''
# w.r.t. x axis in 0
Rx0_prime = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])

# w.r.t. y axis in 0
Ry0_prime = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])

# w.r.t. z axis in 0
Rz0_prime = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])

# given a 6D vector, returns the homogenous transform (rotation matrix + translation vector)
def v2t(v:np.array) -> np.array:
    H = np.eye(4)
    H[0:3, 0:3] = Rx(v[3]) * Ry(v[4]) * Rz(v[5])
    H[0:3, 3] = v[0:3].transpose()
    return H

#given a rotation matrix, returns an unit normalized quaternion (ignore w)
def mat2quat(m:np.array) -> np.array:
    #get element on main diagonal and calculate the trace
    m00, m01, m02 = m[0,0], m[0,1], m[0,2]
    m10, m11, m12 = m[1,0], m[1,1], m[1,2]
    m20, m21, m22 = m[2,0], m[2,1], m[2,2]
    trace = np.trace(m)
    
    #facing any arithmetic issues (avoid division by zero or by a very small number, avoid square root of a negative number)
    if trace > 0:
        div = 2 * math.sqrt(1 + trace)
        q1 = (m21 - m12) / div
        q2 = (m02 - m20) / div
        q3 = (m10 - m01) / div
        q  = np.array([q1, q2, q3])
    elif m00 > m11 and m00 > m22:
        div = 2 * math.sqrt(1 + m00 - m11 - m22)
        q1 = 0.25 * div
        q2 = (m01 + m10) / div
        q3 = (m20 + m02) / div
        q  = np.array([q1, q2, q3])
    elif m11 > m22:
        div = 2 * math.sqrt(1 + m11 - m00 - m22)
        q1 = (m01 + m10) / div
        q2 = 0.25 * div
        q3 = (m12 + m21) / div
        q  = np.array([q1, q2, q3])
    else:
        div = 2 * math.sqrt(1 + m22 - m00 - m11)
        q1 = (m02 + m20) / div
        q2 = (m12 + m21) / div
        q3 = 0.25 * div
        q  = np.array([q1, q2, q3])
    return q

#given a homogeneous matrix, returns the 6D vector
def t2v(m:np.array) -> np.array:
    v = np.zeros(6)
    v[0:3] = m[0:3,3]
    v[3:6] = mat2quat(m[0:3,0:3])
    return v 

#given a 3D vector of angles, returns the skew symmetric matrix
def skew(v:np.array) -> np.array:
    a1, a2, a3 = v[0], v[1], v[2]
    return np.array([[0, -a3, a2], [a3, 0, -a1], [-a2, a1, 0]])

#check if a matrix M is skew-symmetric e.g. if M = -M.transpose
def is_skewed(M:np.array) -> bool:
    Mt = np.transpose(M)
    cmp = np.equal(M, -Mt).tolist()
    return False if False in cmp else True      #return False if there is at least one False in the cmp list, True otherwise

#returns the index referring to the ith measurements file
def get_meas_file_idx(id:int):
    id = str(id)
    nz = 5 - len(id)        #number of zeros needed to complete the file name
    return '0' * nz + id 

#returns the flatten representation of the input 4x4 matrix
def flattenIsometryByColumn(mat:np.array) -> np.array:
    v = np.zeros([12,1])
    v[0:9] = np.reshape(mat[0:3, 0:3].transpose(), [9,1])
    v[9:12] = mat[0:3, 3].reshape([3,1])
    return v

#returns the index in the perturbation vector, that corresponds to a certain pose
def get_pert_pose_idx(pose_id:int):
    return -1 if pose_id > NUM_POSES else pose_id * POSE_DIM

#returns the index in the perturbation vector, that corresponds to a certain landmark
def get_pert_landmark_idx(landmark_id:int):
    return -1 if landmark_id > NUM_LANDMARKS else NUM_POSES * POSE_DIM + landmark_id * LANDMARK_DIM
