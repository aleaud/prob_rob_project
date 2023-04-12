#file paths
DATASET_PATH = "../dataset"
CAMERA_FILE = "camera.dat"
WORLD_FILE = "world.dat"
TRAJ_FILE = "trajectoy.dat"
MEAS_FILE = "meas-{}.dat"
IMG_DIR = "../imgs"

#number of samples and number of measuments (for testing)
N = 10
M = 3

#sizes
NUM_LANDMARKS = 1000
NUM_POSES = 200          
POSE_DIM = 6
LANDMARK_DIM = 3
PROJ_DIM = 2
SYSTEM_SIZE = POSE_DIM * NUM_POSES + LANDMARK_DIM * NUM_LANDMARKS