import numpy as np
from pr_classes import *
from utils import *

#given an Image Point (and its associated world point), returns True if it is not visible by the camera model, False otherwise
def is_not_visible(cam:Camera, pimg:ImagePoint, pw:Landmark=None) -> bool:
    if pw is not None:
        return (pw.z < cam.z_near or pw.z > cam.z_far or pimg.w < 0 or pimg.w > cam.width or pimg.h < 0 or pimg.h > cam.height)
    else: 
        return (pimg.w < 0 or pimg.w > cam.width or pimg.h < 0 or pimg.h > cam.height)
    
'''
Returns the difference between the prediction and the measurement and
the Jacobians  of state w.r.t. error and perturbation

Input:
-Xr: robot pose in the world frame
-Xl: landmark pose in the world frame
-z: observed point to be projected on the image plane
-cam: camera model

Output:
-is_valid: boolean flag telling if the projection is inside the image plane
-pe: 2x1 projection error between prediction and measurement 
-Jr: 2x6 derivative w.r.t. error and perturbation on the pose
-Jl: 2x3 derivative w.r.t. error and perturbation on the landmark
'''
def projection_error_and_jacobian(Xr:RobotPose, Xl:Landmark, z:ImagePoint, cam:Camera):
    #get the 4x4 homogeneous transform 
    Xr = v2t(Xr.get_euclidean_param(3))

    #extract rotational and translational part from inverse transform
    Xr = Xr @ cam.T
    R = Xr[0:3, 0:3].transpose()
    t = -R @ Xr[0:3, 3]
    
    #express landmark in camera frame
    pw = R @ Xl.get_vec3d().transpose() + t
    pcam = cam.K @ pw
    pimg = pcam[0:2] / pcam[2]
    pimg = ImagePoint(z.id, pimg[0], pimg[1])

    #initialize Jacobian of pcam w.r.t. dx
    Jwr = np.zeros([3,6])
    #initialize Jacobian of projection w.r.t. pcam
    Jp = np.zeros([2,3])
    #initialize projection error on image point
    pe = np.zeros([2,1])
    
    if is_not_visible(cam, pimg):
        #print('Point projection is out of image plane. Full-zero matrices are returned.')
        return False, pe, Jwr, Jp

    #compute derivatives: first 3x3 block is -R (and not the identity) because we go from world to camera frame
    #for robot pose
    Jwr[0:3, 0:3] = -R
    Jwr[0:3, 3:6] = R @ skew(Xl.get_vec3d())
    #for landmark
    Jwl = R
    #for projection
    px, py, pz = pcam[0], pcam[1], pcam[2]
    Jp = np.array([[1/pz,  0,  -px/pow(pz,2)],
                    [0,  1/pz, -py/pow(pz,2)]])
    
    #compute projection error and final Jacobians w.r.t. robot and landmarks
    pe = (pimg.get_vec() - z.get_vec()).reshape([-1,1])
    Jr = Jp @ cam.K @ Jwr
    Jl = Jp @ cam.K @ Jwl
    return True, pe, Jr, Jl

'''
Error and jacobian of a measured pose, all poses are in world frame

Input:
-Xi: observing robot pose
-Xj: bserved robot pose
-Z:  relative transform measured between Xi and Xj

Output:
-e: 12x1 error vector is the difference between prediction and measurement
-Ji : 12x6 derivative w.r.t error and perturbation of Xi
-Jj : 12x6 derivative w.r.t error and perturbation of Xj
'''
def pose_error_and_jacobian(Xi:RobotPose, Xj:RobotPose, Z:np.array):
    #get homogeneous form of states
    Xi = v2t(Xi.get_euclidean_param())
    Xj = v2t(Xj.get_euclidean_param())
    
    #extract rotational parts
    Ri = Xi[0:3, 0:3]
    Rj = Xj[0:3, 0:3]
    #extract translational parts 
    ti = Xi[0:3, 3]
    tj = Xj[0:3, 3]
    t_diff = tj - ti
    Ri_t = np.transpose(Ri)

    #partial derivates
    dR_dax = Ri_t @ Rx0_prime @ Rj
    dR_day = Ri_t @ Ry0_prime @ Rj
    dR_daz = Ri_t @ Rz0_prime @ Rj

    #chordal jacobian of h(Xj+dxj,Xi) w.r.t. dxi
    Jj = np.zeros([12,6])
    Jj[0:9, 3] = np.ravel(dR_dax)
    Jj[0:9, 4] = np.ravel(dR_day)
    Jj[0:9, 5] = np.ravel(dR_daz)
    Jj[9:12, 0:3] = Ri_t
    Jj[9:12, 3:6] = Ri_t @ skew(-tj)

    #chordal jacobian of h(Xj+dxj,Xi) w.r.t. dxj => opposite of the previous one!
    Ji = -Jj

    #pose error
    Z_hat = np.eye(4)
    #print("Z_hat shape: {} Z shape: {}".format(Z_hat.shape, Z.shape))
    Z_hat[0:3, 0:3] = Ri_t @ Rj
    Z_hat[0:3, 3] = Ri_t @ t_diff
    e = flattenIsometryByColumn(Z_hat - Z)

    return e, Ji, Jj